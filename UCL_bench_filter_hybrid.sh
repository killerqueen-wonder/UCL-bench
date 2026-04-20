#!/bin/bash

# ============================================
# 法律LLM评估脚本
# 用法: bash run_legal_evaluation.sh --test_model RL2.0.3 --date 0123 --model_path /完整/模型/路径
# ============================================

# 设置错误时退出
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
TEST_MODEL=""
DATE=""
MODEL_PATH=""
# 默认检索路径
RETRIEVE_URL="http://127.0.0.1:8007/retrieve"

# 显示帮助信息
show_help() {
    echo -e "${GREEN}用法: $0 [选项]${NC}"
    echo "选项:"
    echo "  -m, --test_model   测试模型名称 (如: RL2.0.3)"
    echo "  -d, --date         日期标识 (如: 0123)"
    echo "  -p, --model_path   完整模型路径 (必需)"
    echo "                     示例: /F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-2.0.3/global_step_40/actor_merge"
    echo "  -r, --retrieve_url   检索服务地址 (可选，默认: http://127.0.0.1:8007/retrieve)"
    echo "  -h, --help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --test_model RL2.0.3 --date 0123 --model_path /完整/模型/路径"
    echo "  $0 -m RL2.0.3 -d 0123 -p /完整/模型/路径"
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--test_model)
                TEST_MODEL="$2"
                shift 2
                ;;
            -d|--date)
                DATE="$2"
                shift 2
                ;;
            -p|--model_path)
                MODEL_PATH="$2"
                shift 2
                ;;
            -r|--retrieve_url)
                RETRIEVE_URL="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo -e "${RED}错误: 未知参数 '$1'${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查必需参数
    if [ -z "$TEST_MODEL" ]; then
        echo -e "${RED}错误: 必须提供 --test_model 参数${NC}"
        show_help
        exit 1
    fi
    
    if [ -z "$DATE" ]; then
        echo -e "${RED}错误: 必须提供 --date 参数${NC}"
        show_help
        exit 1
    fi
    
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}错误: 必须提供 --model_path 参数${NC}"
        echo -e "${YELLOW}请提供完整的模型路径，例如：${NC}"
        echo -e "${YELLOW}  /F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-2.0.3/global_step_40/actor_merge${NC}"
        show_help
        exit 1
    fi
}

# 验证模型路径
validate_model_path() {
    echo -e "${BLUE}验证模型路径...${NC}"
    
    # 检查路径是否存在
    if [ ! -d "$MODEL_PATH" ] && [ ! -f "$MODEL_PATH" ]; then
        echo -e "${RED}错误: 模型路径不存在或不可访问${NC}"
        echo -e "${RED}路径: $MODEL_PATH${NC}"
        echo -e "${YELLOW}请检查：${NC}"
        echo -e "${YELLOW}1. 路径是否正确${NC}"
        echo -e "${YELLOW}2. 文件/目录权限是否足够${NC}"
        echo -e "${YELLOW}3. 路径是否包含特殊字符${NC}"
        exit 1
    fi
    
    # 显示路径信息
    echo -e "${GREEN}✓ 模型路径验证通过${NC}"
    
    # 检查是否是目录
    if [ -d "$MODEL_PATH" ]; then
        echo -e "${BLUE}模型类型: 目录${NC}"
    elif [ -f "$MODEL_PATH" ]; then
        echo -e "${BLUE}模型类型: 文件${NC}"
        echo -e "${BLUE}文件信息:${NC}"
        ls -la "$MODEL_PATH"
    fi
}

main() {
    parse_args "$@"
    
    FILE_PREFIX="qwen3-8B_${TEST_MODEL}"
    MODEL_RESULT_FILE="${FILE_PREFIX}_eval_result_${DATE}.json"
    MODEL_RESULT_PATH="/data/panghuaiwen/legal_R1/dataset/result/res_result/${MODEL_RESULT_FILE}"
    SCORE_RESULT_FILE="${FILE_PREFIX}_score_${DATE}.json"
    SCORE_RESULT_PATH="/data/panghuaiwen/legal_R1/dataset/result/score_result/${SCORE_RESULT_FILE}"

    # =============== 核心新增：启动额外的 vLLM 评测与总结服务 =============== #
    echo -e "${YELLOW}[系统调度] 正在启动辅助大模型服务...${NC}"
    
    # 启动供 Summary 总结使用的模型 (挂载在显卡 1，端口 8008)
    tmux new -d -s vllm_summarizer "export CUDA_VISIBLE_DEVICES=1; conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model /data/panghuaiwen/legal_R1/model/Qwen/Qwen3-8B --served-model-name Qwen3-8B --port 8008 --gpu-memory-utilization 0.4 --max-model-len 8000"
    
    # 启动供 Eval Judge 评测使用的模型 (挂载在显卡 3，端口 8009)
    tmux new -d -s vllm_evaluator "export CUDA_VISIBLE_DEVICES=3; conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model /data/panghuaiwen/legal_R1/model/Qwen/Qwen2.5-8B-Instruct --served-model-name Qwen2.5-8B-Instruct --port 8009 --gpu-memory-utilization 0.4 --max-model-len 16000"

    echo -e "${YELLOW}等待服务热身 (30s)...${NC}"
    sleep 30

    # =============== 推理阶段（并行） =============== #
    echo -e "${YELLOW}[4/5] 运行并行推理脚本...${NC}"
    
    # 使用 accelerate 多卡并行提速本地生成
    accelerate launch --multi_gpu --num_processes=4 "/data/panghuaiwen/legal_R1/UCL-bench/local_inference_single_round.py" \
        --data_path "/data/panghuaiwen/legal_R1/UCL-bench/dataset/legal_data_sample.json" \
        --model_path "$MODEL_PATH" \
        --result_path "$MODEL_RESULT_PATH" \
        --retriever true \
        --topk 10 \
        --max_turn 12 \
        --retrieve_path "$RETRIEVE_URL" \
        --summary_port 8008
    
    # =============== 评测阶段（并发） =============== #
    echo -e "${YELLOW}[5/5] 运行高并发评估脚本...${NC}"
    python "/data/panghuaiwen/legal_R1/UCL-bench/local_evaluate_thinking.py" \
        --chatgpt_result_path "/data/panghuaiwen/legal_R1/dataset/result/res_result/qwen3_8B_eval_result.json" \
        --model_result_path "$MODEL_RESULT_PATH" \
        --datasource_path "/data/panghuaiwen/legal_R1/UCL-bench/dataset/legal_data_sample.json" \
        --result_path "$SCORE_RESULT_PATH" \
        --judge_port 8009
    
    echo -e "${GREEN}管道全流程执行完毕！${NC}"
}
main "$@"