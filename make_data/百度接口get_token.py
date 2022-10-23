
# encoding:utf-8
import requests

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=EAZij5o5HRTOZl9iVvcNWnuZ&client_secret=msiVrAHHHkNmDkUQy0flUNy0yNkWu0cM'
response = requests.get(host)
if response:
    print(response.json())
    #结果:
    #{'refresh_token': '25.f604271239f7d7c53fff7c099e6cfb44.315360000.1959920243.282335-25595602', 'expires_in': 2592000, 'session_key': '9mzdXvT3erJT+cuyf3hKckX7ovj0GGvF5kdz/zqpwgCbETt9BLPK+HKndJbmJvc8Wrodg+qdS+x060Y+PcqhyTuH200EUw==', 'access_token': '24.e174e7b36d48d2d74590ffcb2a234eda.2592000.1647152243.282335-25595602', 'scope': 'public brain_all_scope brain_colourize brain_stretch_restore brain_dehaze brain_contrast_enhance brain_image_quality_enhance brain_style_trans brain_inpainting brain_image_definition_enhance brain_selfie_anime wise_adapt lebo_resource_base lightservice_public hetu_basic lightcms_map_poi kaidian_kaidian ApsMisTest_Test权限 vis-classify_flower lpq_开放 cop_helloScope ApsMis_fangdi_permission smartapp_snsapi_base smartapp_mapp_dev_manage iop_autocar oauth_tp_app smartapp_smart_game_openapi oauth_sessionkey smartapp_swanid_verify smartapp_opensource_openapi smartapp_opensource_recapi fake_face_detect_开放Scope vis-ocr_虚拟人物助理 idl-video_虚拟人物助理 smartapp_component smartapp_search_plugin avatar_video_test b2b_tp_openapi b2b_tp_openapi_online', 'session_secret': '042d54ed62727e7a8a54183f0783b9f6'}
    #token:
    #24.e174e7b36d48d2d74590ffcb2a234eda.2592000.1647152243.282335-25595602