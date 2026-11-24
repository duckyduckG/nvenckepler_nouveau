/*
 * Kepler GK104/GK106/GK107 MSENC (0x90b7) H.264 OBS Studio encoder
 * using open-source Mesa Gallium VA-API (nouveau) via FFmpeg libavcodec
 * with zero-copy EGL DMA-BUF -> VA-API texture import (encode_texture2).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <obs-avc.h>
#include <obs-module.h>
#include <graphics/graphics.h>
#include <util/bmem.h>
#include <util/darray.h>
#include <util/dstr.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <libdrm/drm_fourcc.h>
#include <va/va.h>
#include <va/va_drmcommon.h>

#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>

#ifndef DEFAULT_KEPLER_LIBVA_DRIVERS_PATH
#define DEFAULT_KEPLER_LIBVA_DRIVERS_PATH ""
#endif

#define KEPLER_TEX_CACHE_MAX 16
#define kepler_log(level, format, ...) blog(level, "[nvenckepler-vaapi] " format, ##__VA_ARGS__)

static pthread_mutex_t g_env_mutex = PTHREAD_MUTEX_INITIALIZER;

struct kepler_va_tex_cache {
	GLuint y_id;
	GLuint uv_id;
	VASurfaceID va_surf;
};

struct kepler_vaapi_encoder {
	obs_encoder_t *encoder;

	AVBufferRef *hw_device_ctx;
	AVBufferRef *hw_frames_ctx;
	AVCodecContext *ctx;
	AVFrame *sw_frame;
	AVFrame *hw_frame;
	AVPacket *pkt;

	uint32_t width;
	uint32_t height;
	int64_t total_frames;

	uint8_t *header;
	size_t header_size;
	uint8_t *sei;
	size_t sei_size;

	DARRAY(uint8_t) packet_buf;

	/* Zero-copy EGL DMA-BUF -> VA-API surface cache */
	struct kepler_va_tex_cache tex_cache[KEPLER_TEX_CACHE_MAX];
	size_t tex_cache_count;
	bool logged_zerocopy;

	PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR;
	PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR;
	PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC eglExportDMABUFImageQueryMESA;
	PFNEGLEXPORTDMABUFIMAGEMESAPROC eglExportDMABUFImageMESA;
};

static bool file_exists(const char *path)
{
	struct stat st;
	return path && path[0] && stat(path, &st) == 0;
}

static bool detect_nouveau_render_node(char *out, size_t out_size)
{
	DIR *dir = opendir("/sys/class/drm");
	if (dir) {
		struct dirent *de;
		while ((de = readdir(dir)) != NULL) {
			if (strncmp(de->d_name, "renderD", 7) != 0)
				continue;

			char link_path[512];
			char target[256];
			snprintf(link_path, sizeof(link_path), "/sys/class/drm/%.64s/device/driver", de->d_name);
			ssize_t len = readlink(link_path, target, sizeof(target) - 1);
			if (len <= 0)
				continue;
			target[len] = '\0';

			const char *base = strrchr(target, '/');
			base = base ? base + 1 : target;
			if (strcmp(base, "nouveau") == 0) {
				snprintf(out, out_size, "/dev/dri/%.64s", de->d_name);
				closedir(dir);
				return true;
			}
		}
		closedir(dir);
	}

	if (file_exists("/dev/dri/by-path/pci-0000:01:00.0-render")) {
		snprintf(out, out_size, "/dev/dri/by-path/pci-0000:01:00.0-render");
		return true;
	}
	return false;
}

static const char *default_vaapi_driver_dir(void)
{
	const char *env_path = getenv("LIBVA_DRIVERS_PATH");
	if (env_path && env_path[0])
		return env_path;
	if (DEFAULT_KEPLER_LIBVA_DRIVERS_PATH[0]) {
		char probe[512];
		snprintf(probe, sizeof(probe), "%s/nouveau_drv_video.so", DEFAULT_KEPLER_LIBVA_DRIVERS_PATH);
		if (file_exists(probe))
			return DEFAULT_KEPLER_LIBVA_DRIVERS_PATH;
	}
	return "";
}

static int create_vaapi_device_scoped(AVBufferRef **out_ctx, const char *device_path, const char *driver_dir)
{
	pthread_mutex_lock(&g_env_mutex);

	const char *old_path_env = getenv("LIBVA_DRIVERS_PATH");
	const char *old_name_env = getenv("LIBVA_DRIVER_NAME");
	char *saved_path = old_path_env ? bstrdup(old_path_env) : NULL;
	char *saved_name = old_name_env ? bstrdup(old_name_env) : NULL;

	if (driver_dir && driver_dir[0])
		setenv("LIBVA_DRIVERS_PATH", driver_dir, 1);
	setenv("LIBVA_DRIVER_NAME", "nouveau", 1);

	int ret = av_hwdevice_ctx_create(out_ctx, AV_HWDEVICE_TYPE_VAAPI, device_path, NULL, 0);

	if (saved_path) {
		setenv("LIBVA_DRIVERS_PATH", saved_path, 1);
		bfree(saved_path);
	} else {
		unsetenv("LIBVA_DRIVERS_PATH");
	}
	if (saved_name) {
		setenv("LIBVA_DRIVER_NAME", saved_name, 1);
		bfree(saved_name);
	} else {
		unsetenv("LIBVA_DRIVER_NAME");
	}

	pthread_mutex_unlock(&g_env_mutex);
	return ret;
}

bool kepler_vaapi_supported(void)
{
	char dev[128];
	if (!detect_nouveau_render_node(dev, sizeof(dev))) {
		kepler_log(LOG_INFO, "No Nouveau DRM render node found");
		return false;
	}

	const AVCodec *codec = avcodec_find_encoder_by_name("h264_vaapi");
	if (!codec) {
		kepler_log(LOG_WARNING, "FFmpeg h264_vaapi encoder not available");
		return false;
	}

	AVBufferRef *hw = NULL;
	int ret = create_vaapi_device_scoped(&hw, dev, default_vaapi_driver_dir());
	if (ret < 0 || !hw) {
		kepler_log(LOG_WARNING, "Failed to open VA-API on %s (err=%d)", dev, ret);
		return false;
	}
	av_buffer_unref(&hw);

	kepler_log(LOG_INFO, "Kepler Nouveau VA-API H.264 encoder available on %s", dev);
	return true;
}

static const char *kepler_get_name(void *type_data)
{
	UNUSED_PARAMETER(type_data);
	return "NVIDIA NVENC H.264 (Kepler Nouveau)";
}

static void kepler_destroy_tex_cache(struct kepler_vaapi_encoder *enc)
{
	if (!enc->hw_device_ctx || enc->tex_cache_count == 0)
		return;
	AVHWDeviceContext *dctx = (AVHWDeviceContext *)enc->hw_device_ctx->data;
	AVVAAPIDeviceContext *vactx = (AVVAAPIDeviceContext *)dctx->hwctx;
	for (size_t i = 0; i < enc->tex_cache_count; i++) {
		if (enc->tex_cache[i].va_surf != VA_INVALID_SURFACE) {
			vaDestroySurfaces(vactx->display, &enc->tex_cache[i].va_surf, 1);
			enc->tex_cache[i].va_surf = VA_INVALID_SURFACE;
		}
	}
	enc->tex_cache_count = 0;
}

static void kepler_destroy(void *data)
{
	struct kepler_vaapi_encoder *enc = data;
	if (!enc)
		return;

	kepler_destroy_tex_cache(enc);

	if (enc->pkt)
		av_packet_free(&enc->pkt);
	if (enc->hw_frame)
		av_frame_free(&enc->hw_frame);
	if (enc->sw_frame)
		av_frame_free(&enc->sw_frame);
	if (enc->ctx)
		avcodec_free_context(&enc->ctx);
	if (enc->hw_frames_ctx)
		av_buffer_unref(&enc->hw_frames_ctx);
	if (enc->hw_device_ctx)
		av_buffer_unref(&enc->hw_device_ctx);

	bfree(enc->header);
	bfree(enc->sei);
	da_free(enc->packet_buf);
	bfree(enc);
}

static void *kepler_create(obs_data_t *settings, obs_encoder_t *encoder)
{
	struct kepler_vaapi_encoder *enc = bzalloc(sizeof(*enc));
	enc->encoder = encoder;

	video_t *video = obs_encoder_video(encoder);
	const struct video_output_info *voi = video_output_get_info(video);

	enc->width = obs_encoder_get_width(encoder);
	enc->height = obs_encoder_get_height(encoder);

	const char *device = obs_data_get_string(settings, "vaapi_device");
	char auto_dev[128] = {0};
	if (!device || !device[0]) {
		if (!detect_nouveau_render_node(auto_dev, sizeof(auto_dev))) {
			kepler_log(LOG_ERROR, "No Nouveau render node detected");
			goto fail;
		}
		device = auto_dev;
	}

	const char *driver_dir = obs_data_get_string(settings, "vaapi_driver_path");
	if (!driver_dir || !driver_dir[0])
		driver_dir = default_vaapi_driver_dir();

	int ret = create_vaapi_device_scoped(&enc->hw_device_ctx, device, driver_dir);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		kepler_log(LOG_ERROR, "av_hwdevice_ctx_create(%s) failed: %s", device, errbuf);
		goto fail;
	}

	enc->hw_frames_ctx = av_hwframe_ctx_alloc(enc->hw_device_ctx);
	if (!enc->hw_frames_ctx)
		goto fail;

	AVHWFramesContext *fctx = (AVHWFramesContext *)enc->hw_frames_ctx->data;
	fctx->format = AV_PIX_FMT_VAAPI;
	fctx->sw_format = AV_PIX_FMT_NV12;
	fctx->width = (int)enc->width;
	fctx->height = (int)enc->height;
	fctx->initial_pool_size = 8;

	ret = av_hwframe_ctx_init(enc->hw_frames_ctx);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		kepler_log(LOG_ERROR, "av_hwframe_ctx_init failed: %s", errbuf);
		goto fail;
	}

	const AVCodec *codec = avcodec_find_encoder_by_name("h264_vaapi");
	if (!codec) {
		kepler_log(LOG_ERROR, "h264_vaapi codec not found");
		goto fail;
	}

	enc->ctx = avcodec_alloc_context3(codec);
	if (!enc->ctx)
		goto fail;

	const char *rc = obs_data_get_string(settings, "rate_control");
	int bitrate_kbps = (int)obs_data_get_int(settings, "bitrate");
	int max_bitrate_kbps = (int)obs_data_get_int(settings, "max_bitrate");
	int cqp = (int)obs_data_get_int(settings, "cqp");
	int keyint_sec = (int)obs_data_get_int(settings, "keyint_sec");
	const char *profile = obs_data_get_string(settings, "profile");
	const char *preset = obs_data_get_string(settings, "preset");

	if (!rc || !rc[0])
		rc = "CBR";
	if (bitrate_kbps < 50)
		bitrate_kbps = 6000;
	if (max_bitrate_kbps < bitrate_kbps)
		max_bitrate_kbps = bitrate_kbps;
	if (cqp < 1 || cqp > 51)
		cqp = 26;

	enc->ctx->width = (int)enc->width;
	enc->ctx->height = (int)enc->height;
	enc->ctx->pix_fmt = AV_PIX_FMT_VAAPI;
	enc->ctx->hw_frames_ctx = av_buffer_ref(enc->hw_frames_ctx);
	enc->ctx->time_base = (AVRational){(int)voi->fps_den, (int)voi->fps_num};
	enc->ctx->framerate = (AVRational){(int)voi->fps_num, (int)voi->fps_den};
	enc->ctx->max_b_frames = 0; /* Kepler MSENC P-frame GOP */
	enc->ctx->gop_size = keyint_sec > 0 ? (int)((int64_t)keyint_sec * voi->fps_num / voi->fps_den) : 250;
	enc->ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

	/* Map quality preset to VA-API compression_level (1=Quality, 2=Balanced, 3=Performance) */
	if (preset && strcasecmp(preset, "quality") == 0)
		enc->ctx->compression_level = 1;
	else if (preset && strcasecmp(preset, "balanced") == 0)
		enc->ctx->compression_level = 2;
	else
		enc->ctx->compression_level = 3;

	/* Color metadata */
	enc->ctx->color_range = (voi->range == VIDEO_RANGE_FULL) ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
	switch (voi->colorspace) {
	case VIDEO_CS_601:
		enc->ctx->colorspace = AVCOL_SPC_SMPTE170M;
		enc->ctx->color_primaries = AVCOL_PRI_SMPTE170M;
		enc->ctx->color_trc = AVCOL_TRC_SMPTE170M;
		break;
	case VIDEO_CS_709:
	default:
		enc->ctx->colorspace = AVCOL_SPC_BT709;
		enc->ctx->color_primaries = AVCOL_PRI_BT709;
		enc->ctx->color_trc = AVCOL_TRC_BT709;
		break;
	}

	if (profile && strcmp(profile, "baseline") == 0)
		enc->ctx->profile = AV_PROFILE_H264_CONSTRAINED_BASELINE;
	else if (profile && strcmp(profile, "main") == 0)
		enc->ctx->profile = AV_PROFILE_H264_MAIN;
	else
		enc->ctx->profile = AV_PROFILE_H264_HIGH;

	AVDictionary *opts = NULL;
	if (strcasecmp(rc, "CQP") == 0) {
		av_dict_set(&opts, "rc_mode", "CQP", 0);
		av_dict_set_int(&opts, "qp", cqp, 0);
		enc->ctx->global_quality = cqp;
	} else if (strcasecmp(rc, "VBR") == 0) {
		av_dict_set(&opts, "rc_mode", "VBR", 0);
		enc->ctx->bit_rate = (int64_t)bitrate_kbps * 1000;
		enc->ctx->rc_max_rate = (int64_t)max_bitrate_kbps * 1000;
		enc->ctx->rc_buffer_size = (int)(max_bitrate_kbps * 1000);
	} else {
		av_dict_set(&opts, "rc_mode", "CBR", 0);
		enc->ctx->bit_rate = (int64_t)bitrate_kbps * 1000;
		enc->ctx->rc_max_rate = (int64_t)bitrate_kbps * 1000;
		enc->ctx->rc_buffer_size = (int)(bitrate_kbps * 1000);
	}

	ret = avcodec_open2(enc->ctx, codec, &opts);
	av_dict_free(&opts);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		kepler_log(LOG_ERROR, "avcodec_open2(h264_vaapi) failed: %s", errbuf);
		goto fail;
	}

	if (enc->ctx->extradata && enc->ctx->extradata_size > 0) {
		enc->header = bmemdup(enc->ctx->extradata, (size_t)enc->ctx->extradata_size);
		enc->header_size = (size_t)enc->ctx->extradata_size;
	}

	enc->sw_frame = av_frame_alloc();
	enc->hw_frame = av_frame_alloc();
	enc->pkt = av_packet_alloc();
	if (!enc->sw_frame || !enc->hw_frame || !enc->pkt)
		goto fail;

	enc->sw_frame->format = AV_PIX_FMT_NV12;
	enc->sw_frame->width = (int)enc->width;
	enc->sw_frame->height = (int)enc->height;

	enc->eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
	enc->eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
	enc->eglExportDMABUFImageQueryMESA =
		(PFNEGLEXPORTDMABUFIMAGEQUERYMESAPROC)eglGetProcAddress("eglExportDMABUFImageQueryMESA");
	enc->eglExportDMABUFImageMESA =
		(PFNEGLEXPORTDMABUFIMAGEMESAPROC)eglGetProcAddress("eglExportDMABUFImageMESA");

	kepler_log(LOG_INFO,
		   "Initialized Kepler MSENC H.264 encoder: %ux%u @ %u/%u fps, rc=%s, bitrate=%dkbps, qp=%d, preset=%s, dev=%s",
		   enc->width, enc->height, voi->fps_num, voi->fps_den, rc, bitrate_kbps, cqp,
		   (preset && preset[0]) ? preset : "performance", device);
	return enc;

fail:
	kepler_destroy(enc);
	return NULL;
}

static bool kepler_update(void *data, obs_data_t *settings)
{
	struct kepler_vaapi_encoder *enc = data;
	if (!enc || !enc->ctx)
		return false;

	const char *rc = obs_data_get_string(settings, "rate_control");
	int bitrate_kbps = (int)obs_data_get_int(settings, "bitrate");
	int max_bitrate_kbps = (int)obs_data_get_int(settings, "max_bitrate");
	if (bitrate_kbps >= 50 && (!rc || strcasecmp(rc, "CQP") != 0)) {
		enc->ctx->bit_rate = (int64_t)bitrate_kbps * 1000;
		if (rc && strcasecmp(rc, "VBR") == 0 && max_bitrate_kbps >= bitrate_kbps)
			enc->ctx->rc_max_rate = (int64_t)max_bitrate_kbps * 1000;
		else
			enc->ctx->rc_max_rate = (int64_t)bitrate_kbps * 1000;
	}
	return true;
}

static bool kepler_submit_and_receive(struct kepler_vaapi_encoder *enc, AVFrame *hw_frame,
				      struct encoder_packet *packet, bool *received_packet)
{
	int ret = avcodec_send_frame(enc->ctx, hw_frame);
	if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		kepler_log(LOG_ERROR, "avcodec_send_frame failed: %s", errbuf);
		return false;
	}

	av_packet_unref(enc->pkt);
	ret = avcodec_receive_packet(enc->ctx, enc->pkt);
	if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
		return true;
	} else if (ret < 0) {
		char errbuf[128];
		av_strerror(ret, errbuf, sizeof(errbuf));
		kepler_log(LOG_ERROR, "avcodec_receive_packet failed: %s", errbuf);
		return false;
	}

	if (!enc->header && enc->pkt->size > 0) {
		uint8_t *new_pkt = NULL;
		size_t new_size = 0;
		obs_extract_avc_headers(enc->pkt->data, (size_t)enc->pkt->size, &new_pkt, &new_size, &enc->header,
					&enc->header_size, &enc->sei, &enc->sei_size);
		if (new_pkt && new_size > 0) {
			da_copy_array(enc->packet_buf, new_pkt, new_size);
			bfree(new_pkt);
		} else {
			da_copy_array(enc->packet_buf, enc->pkt->data, (size_t)enc->pkt->size);
		}
	} else {
		da_copy_array(enc->packet_buf, enc->pkt->data, (size_t)enc->pkt->size);
	}

	packet->data = enc->packet_buf.array;
	packet->size = enc->packet_buf.num;
	packet->type = OBS_ENCODER_VIDEO;
	packet->pts = enc->pkt->pts;
	packet->dts = enc->pkt->dts;
	packet->keyframe = ((enc->pkt->flags & AV_PKT_FLAG_KEY) != 0) ||
			   obs_avc_keyframe(packet->data, packet->size);
	packet->priority = packet->keyframe ? OBS_NAL_PRIORITY_HIGHEST : OBS_NAL_PRIORITY_HIGH;
	packet->drop_priority = packet->priority;

	enc->total_frames++;
	*received_packet = true;
	return true;
}

static bool kepler_encode(void *data, struct encoder_frame *frame, struct encoder_packet *packet,
			  bool *received_packet)
{
	struct kepler_vaapi_encoder *enc = data;
	*received_packet = false;

	enc->sw_frame->data[0] = frame->data[0];
	enc->sw_frame->data[1] = frame->data[1];
	enc->sw_frame->linesize[0] = (int)frame->linesize[0];
	enc->sw_frame->linesize[1] = (int)frame->linesize[1];

	av_frame_unref(enc->hw_frame);
	if (av_hwframe_get_buffer(enc->hw_frames_ctx, enc->hw_frame, 0) < 0) {
		kepler_log(LOG_ERROR, "av_hwframe_get_buffer failed");
		return false;
	}
	if (av_hwframe_transfer_data(enc->hw_frame, enc->sw_frame, 0) < 0) {
		kepler_log(LOG_ERROR, "av_hwframe_transfer_data failed");
		return false;
	}
	enc->hw_frame->pts = frame->pts;

	return kepler_submit_and_receive(enc, enc->hw_frame, packet, received_packet);
}

static bool export_gl_tex_dmabuf(struct kepler_vaapi_encoder *enc, EGLDisplay dpy, EGLContext ctx,
				 GLuint tex_id, int *out_fd, int *out_stride, int *out_offset,
				 uint64_t *out_mod, int *out_fourcc)
{
	EGLImageKHR img = enc->eglCreateImageKHR(dpy, ctx, EGL_GL_TEXTURE_2D_KHR,
						 (EGLClientBuffer)(uintptr_t)tex_id, NULL);
	if (img == EGL_NO_IMAGE_KHR)
		return false;

	int fourcc = 0, num_planes = 0;
	EGLuint64KHR mod = DRM_FORMAT_MOD_INVALID;
	bool ok = enc->eglExportDMABUFImageQueryMESA(dpy, img, &fourcc, &num_planes, &mod) && num_planes >= 1;
	int fd = -1;
	EGLint stride = 0, offset = 0;
	if (ok)
		ok = enc->eglExportDMABUFImageMESA(dpy, img, &fd, &stride, &offset) && fd >= 0;

	enc->eglDestroyImageKHR(dpy, img);
	if (!ok) {
		if (fd >= 0)
			close(fd);
		return false;
	}

	*out_fd = fd;
	*out_stride = stride;
	*out_offset = offset;
	*out_mod = (uint64_t)mod;
	*out_fourcc = fourcc;
	return true;
}

static VASurfaceID kepler_get_or_import_va_surface(struct kepler_vaapi_encoder *enc,
						   GLuint y_id, GLuint uv_id)
{
	for (size_t i = 0; i < enc->tex_cache_count; i++) {
		if (enc->tex_cache[i].y_id == y_id && enc->tex_cache[i].uv_id == uv_id)
			return enc->tex_cache[i].va_surf;
	}

	if (!enc->eglCreateImageKHR || !enc->eglDestroyImageKHR ||
	    !enc->eglExportDMABUFImageQueryMESA || !enc->eglExportDMABUFImageMESA)
		return VA_INVALID_SURFACE;

	EGLDisplay dpy = eglGetCurrentDisplay();
	EGLContext ctx = eglGetCurrentContext();
	if (dpy == EGL_NO_DISPLAY || ctx == EGL_NO_CONTEXT)
		return VA_INVALID_SURFACE;

	int y_fd = -1, uv_fd = -1;
	int y_stride = 0, y_off = 0, y_fourcc = 0;
	int uv_stride = 0, uv_off = 0, uv_fourcc = 0;
	uint64_t y_mod = 0, uv_mod = 0;

	if (!export_gl_tex_dmabuf(enc, dpy, ctx, y_id, &y_fd, &y_stride, &y_off, &y_mod, &y_fourcc))
		return VA_INVALID_SURFACE;
	if (!export_gl_tex_dmabuf(enc, dpy, ctx, uv_id, &uv_fd, &uv_stride, &uv_off, &uv_mod, &uv_fourcc)) {
		close(y_fd);
		return VA_INVALID_SURFACE;
	}

	VADRMPRIMESurfaceDescriptor desc = {0};
	desc.fourcc = VA_FOURCC_NV12;
	desc.width = enc->width;
	desc.height = enc->height;
	desc.num_objects = 2;
	desc.objects[0].fd = y_fd;
	desc.objects[0].size = (uint32_t)y_stride * enc->height;
	desc.objects[0].drm_format_modifier = y_mod;
	desc.objects[1].fd = uv_fd;
	desc.objects[1].size = (uint32_t)uv_stride * (enc->height / 2);
	desc.objects[1].drm_format_modifier = uv_mod;

	desc.num_layers = 2;
	desc.layers[0].drm_format = y_fourcc ? (uint32_t)y_fourcc : DRM_FORMAT_R8;
	desc.layers[0].num_planes = 1;
	desc.layers[0].object_index[0] = 0;
	desc.layers[0].offset[0] = (uint32_t)y_off;
	desc.layers[0].pitch[0] = (uint32_t)y_stride;

	desc.layers[1].drm_format = uv_fourcc ? (uint32_t)uv_fourcc : DRM_FORMAT_GR88;
	desc.layers[1].num_planes = 1;
	desc.layers[1].object_index[0] = 1;
	desc.layers[1].offset[0] = (uint32_t)uv_off;
	desc.layers[1].pitch[0] = (uint32_t)uv_stride;

	VASurfaceAttrib attribs[2] = {0};
	attribs[0].type = VASurfaceAttribMemoryType;
	attribs[0].flags = VA_SURFACE_ATTRIB_SETTABLE;
	attribs[0].value.type = VAGenericValueTypeInteger;
	attribs[0].value.value.i = VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2;
	attribs[1].type = VASurfaceAttribExternalBufferDescriptor;
	attribs[1].flags = VA_SURFACE_ATTRIB_SETTABLE;
	attribs[1].value.type = VAGenericValueTypePointer;
	attribs[1].value.value.p = &desc;

	AVHWDeviceContext *dctx = (AVHWDeviceContext *)enc->hw_device_ctx->data;
	AVVAAPIDeviceContext *vactx = (AVVAAPIDeviceContext *)dctx->hwctx;
	VASurfaceID surf = VA_INVALID_SURFACE;
	VAStatus st = vaCreateSurfaces(vactx->display, VA_RT_FORMAT_YUV420,
				       enc->width, enc->height, &surf, 1, attribs, 2);
	close(y_fd);
	close(uv_fd);

	if (st != VA_STATUS_SUCCESS || surf == VA_INVALID_SURFACE) {
		kepler_log(LOG_ERROR, "vaCreateSurfaces(DRM_PRIME_2) failed: %d", st);
		return VA_INVALID_SURFACE;
	}

	if (enc->tex_cache_count < KEPLER_TEX_CACHE_MAX) {
		enc->tex_cache[enc->tex_cache_count++] = (struct kepler_va_tex_cache){
			.y_id = y_id, .uv_id = uv_id, .va_surf = surf,
		};
	}
	if (!enc->logged_zerocopy) {
		kepler_log(LOG_INFO, "Zero-copy EGL DMA-BUF -> VA-API texture path ACTIVE (%ux%u, mod=0x%llx)",
			   enc->width, enc->height, (unsigned long long)y_mod);
		enc->logged_zerocopy = true;
	}
	return surf;
}

static void kepler_noop_free(void *opaque, uint8_t *data)
{
	UNUSED_PARAMETER(opaque);
	UNUSED_PARAMETER(data);
}

static bool kepler_encode_texture2(void *data, struct encoder_texture *texture, int64_t pts,
				   uint64_t lock_key, uint64_t *next_key,
				   struct encoder_packet *packet, bool *received_packet)
{
	struct kepler_vaapi_encoder *enc = data;
	*received_packet = false;
	*next_key = lock_key;

	if (!texture || !texture->tex[0] || !texture->tex[1]) {
		kepler_log(LOG_ERROR, "encode_texture2: missing NV12 texture planes");
		return false;
	}

	obs_enter_graphics();
	GLuint y_id = *(GLuint *)gs_texture_get_obj(texture->tex[0]);
	GLuint uv_id = *(GLuint *)gs_texture_get_obj(texture->tex[1]);
	VASurfaceID va_surf = kepler_get_or_import_va_surface(enc, y_id, uv_id);
	glFlush();
	obs_leave_graphics();

	if (va_surf == VA_INVALID_SURFACE)
		return false;

	av_frame_unref(enc->hw_frame);
	enc->hw_frame->format = AV_PIX_FMT_VAAPI;
	enc->hw_frame->width = (int)enc->width;
	enc->hw_frame->height = (int)enc->height;
	enc->hw_frame->hw_frames_ctx = av_buffer_ref(enc->hw_frames_ctx);
	enc->hw_frame->buf[0] = av_buffer_create((uint8_t *)(uintptr_t)va_surf, sizeof(VASurfaceID),
						 kepler_noop_free, NULL, 0);
	enc->hw_frame->data[3] = (uint8_t *)(uintptr_t)va_surf;
	enc->hw_frame->pts = pts;

	return kepler_submit_and_receive(enc, enc->hw_frame, packet, received_packet);
}

static bool kepler_get_extra_data(void *data, uint8_t **extra_data, size_t *size)
{
	struct kepler_vaapi_encoder *enc = data;
	if (!enc->header || enc->header_size == 0)
		return false;
	*extra_data = enc->header;
	*size = enc->header_size;
	return true;
}

static bool kepler_get_sei_data(void *data, uint8_t **sei_data, size_t *size)
{
	struct kepler_vaapi_encoder *enc = data;
	if (!enc->sei || enc->sei_size == 0)
		return false;
	*sei_data = enc->sei;
	*size = enc->sei_size;
	return true;
}

static void kepler_get_video_info(void *data, struct video_scale_info *info)
{
	UNUSED_PARAMETER(data);
	info->format = VIDEO_FORMAT_NV12;
}

static void kepler_defaults(obs_data_t *settings)
{
	char dev[128] = {0};
	detect_nouveau_render_node(dev, sizeof(dev));

	obs_data_set_default_string(settings, "rate_control", "CBR");
	obs_data_set_default_int(settings, "bitrate", 6000);
	obs_data_set_default_int(settings, "max_bitrate", 9000);
	obs_data_set_default_int(settings, "cqp", 26);
	obs_data_set_default_int(settings, "keyint_sec", 2);
	obs_data_set_default_string(settings, "preset", "performance");
	obs_data_set_default_string(settings, "profile", "high");
	obs_data_set_default_string(settings, "vaapi_device", dev);
	obs_data_set_default_string(settings, "vaapi_driver_path", default_vaapi_driver_dir());
}

static bool kepler_rc_modified(obs_properties_t *ppts, obs_property_t *p, obs_data_t *settings)
{
	UNUSED_PARAMETER(p);
	const char *rc = obs_data_get_string(settings, "rate_control");
	bool is_cqp = strcasecmp(rc, "CQP") == 0;
	bool is_vbr = strcasecmp(rc, "VBR") == 0;

	obs_property_set_visible(obs_properties_get(ppts, "bitrate"), !is_cqp);
	obs_property_set_visible(obs_properties_get(ppts, "max_bitrate"), is_vbr);
	obs_property_set_visible(obs_properties_get(ppts, "cqp"), is_cqp);
	return true;
}

static obs_properties_t *kepler_properties(void *unused)
{
	UNUSED_PARAMETER(unused);
	obs_properties_t *props = obs_properties_create();
	obs_property_t *p;

	p = obs_properties_add_list(props, "rate_control", "Rate Control", OBS_COMBO_TYPE_LIST,
				    OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p, "CBR", "CBR");
	obs_property_list_add_string(p, "VBR", "VBR");
	obs_property_list_add_string(p, "CQP", "CQP");
	obs_property_set_modified_callback(p, kepler_rc_modified);

	p = obs_properties_add_int(props, "bitrate", "Bitrate", 50, 100000, 50);
	obs_property_int_set_suffix(p, " Kbps");

	p = obs_properties_add_int(props, "max_bitrate", "Max Bitrate", 50, 150000, 50);
	obs_property_int_set_suffix(p, " Kbps");

	obs_properties_add_int(props, "cqp", "CQ Level", 1, 51, 1);

	p = obs_properties_add_int(props, "keyint_sec", "Keyframe Interval (0=auto)", 0, 10, 1);
	obs_property_int_set_suffix(p, " s");

	p = obs_properties_add_list(props, "preset", "Preset", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p, "Performance (Fastest)", "performance");
	obs_property_list_add_string(p, "Balanced", "balanced");
	obs_property_list_add_string(p, "Quality (Max Compression)", "quality");

	p = obs_properties_add_list(props, "profile", "Profile", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(p, "high", "high");
	obs_property_list_add_string(p, "main", "main");
	obs_property_list_add_string(p, "baseline", "baseline");

	obs_properties_add_text(props, "vaapi_device", "DRM Render Node (blank = auto)", OBS_TEXT_DEFAULT);
	obs_properties_add_text(props, "vaapi_driver_path", "Mesa VA-API Driver Path (blank = system)", OBS_TEXT_DEFAULT);

	return props;
}

static struct obs_encoder_info kepler_vaapi_h264_info = {
	.id = "kepler_msenc_h264",
	.type = OBS_ENCODER_VIDEO,
	.codec = "h264",
	.caps = OBS_ENCODER_CAP_PASS_TEXTURE | OBS_ENCODER_CAP_DYN_BITRATE,
	.get_name = kepler_get_name,
	.create = kepler_create,
	.destroy = kepler_destroy,
	.update = kepler_update,
	.encode = kepler_encode,
	.encode_texture2 = kepler_encode_texture2,
	.get_defaults = kepler_defaults,
	.get_properties = kepler_properties,
	.get_extra_data = kepler_get_extra_data,
	.get_sei_data = kepler_get_sei_data,
	.get_video_info = kepler_get_video_info,
};

void obs_kepler_vaapi_load(void)
{
	obs_register_encoder(&kepler_vaapi_h264_info);
}

void obs_kepler_vaapi_unload(void) {}
