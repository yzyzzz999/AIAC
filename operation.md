conda activate AIAC
1. 启动can服务
cd can_service
./can_setup.sh
python can0_service_v1.3.6.py
2. 启动视觉感知服务
cd 0612_seatfixed
./scripts/start.sh
3. 视觉感知服务启动完成后
cd vehicle_runtime_package_vnext_budian
./start_pmv_service.sh
4. 启动推荐服务
cd sls_recommendation/src
python -m src.main_controller --shadow-mode --model-package-mode legacy

