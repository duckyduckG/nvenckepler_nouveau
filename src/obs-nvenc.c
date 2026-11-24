#include <obs-module.h>

#include "obs-nvenc.h"

#ifdef ENABLE_NOUVEAU_VAAPI
extern bool kepler_vaapi_supported(void);
extern void obs_kepler_vaapi_load(void);
extern void obs_kepler_vaapi_unload(void);
#endif

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("nvenckepler", "en-US")

MODULE_EXPORT const char *obs_module_description(void)
{
	return "NVIDIA Kepler Encoder (NVENC / Nouveau MSENC VA-API) Plugin";
}

bool obs_module_load(void)
{
#ifdef ENABLE_NOUVEAU_VAAPI
	if (kepler_vaapi_supported()) {
		obs_kepler_vaapi_load();
		return true;
	}
	blog(LOG_INFO, "Kepler Nouveau VA-API encoder not available");
	return false;
#else
	if (!nvenc_supported()) {
		blog(LOG_INFO, "NVENC not supported");
		return false;
	}

	obs_nvenc_load();
	obs_cuda_load();

	return true;
#endif
}

void obs_module_unload(void)
{
#ifdef ENABLE_NOUVEAU_VAAPI
	obs_kepler_vaapi_unload();
#else
	obs_cuda_unload();
	obs_nvenc_unload();
#endif
}
