#!/bin/bash

# ============================================
# 法律LLM评估脚本
# 用法: bash run_legal_evaluation.sh --test_model RL2.0.3 --date 0123
# ============================================

# 设置错误时退出
set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
TEST_MODEL=""
DATE=""

# 显示帮助信息
show_help() {
    echo -e "${GREEN}用法: $0 [选项]${NC}"
    echo "选项:"
    echo "  -m, --test_model   测试模型名称 (默认: 必须提供)"
    echo "  -d, --date         日期标识 (默认: 必须提供)"
    echo "  -h, --help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --test_model RL2.0.3 --date 0123"
    echo "  $0 -m RL2.0.3 -d 0123"
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
    if [ -z "$TEST_MODEL" ] || [ -z "$DATE" ]; then
        echo -e "${RED}错误: 必须提供 --test_model 和 --date 参数${NC}"
        show_help
        exit 1
    fi
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}开始法律LLM评估流程${NC}"
    echo -e "${GREEN}模型: $TEST_MODEL${NC}"
    echo -e "${GREEN}日期: $DATE${NC}"
    echo -e "${GREEN}============================================${NC}"
    
    # 步骤1: 激活conda环境
    echo -e "${YELLOW}[1/5] 激活conda环境...${NC}"
    conda activate searchr1_new
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 无法激活conda环境 'searchr1_new'${NC}"
        exit 1
    fi
    
    # 步骤2: 切换到项目目录并拉取最新代码
    echo -e "${YELLOW}[2/5] 切换到项目目录并更新代码...${NC}"
    cd /caizhenyang/panghuaiwen/legal_LLM/UCL-bench
    git pull origin main
    
    # 步骤3: 生成动态文件名
    echo -e "${YELLOW}[3/5] 生成输出文件名...${NC}"
    
    # 中间结果文件（推理结果）
    MODEL_RESULT_FILE="qwen3-8B_${TEST_MODEL}_eval_result_${DATE}.json"
    MODEL_RESULT_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/res_result/${MODEL_RESULT_FILE}"
    
    # 最终评分结果文件
    SCORE_RESULT_FILE="qwen3-8B_${TEST_MODEL}_score_${DATE}.json"
    SCORE_RESULT_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/score_result/${SCORE_RESULT_FILE}"
    
    echo "推理结果文件: $MODEL_RESULT_FILE"
    echo "评分结果文件: $SCORE_RESULT_FILE"
    
    # 步骤4: 运行推理脚本
    echo -e "${YELLOW}[4/5] 运行推理脚本...${NC}"
    python "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/UCL-bench/local_inference_single_round.py" \
        --data_path "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/UCL-bench/dataset/legal_data_sample.json" \
        --model_path "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-${TEST_MODEL}/global_step_40/actor_merge" \
        --result_path "$MODEL_RESULT_PATH" \
        --model_name "no_GPT" \
        --retriever true \
        --topk 10 \
        --max_turn 4 \
        --retrieve_path "http://127.0.0.1:8006/retrieve"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 推理脚本执行失败${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}推理完成，结果保存在: $MODEL_RESULT_PATH${NC}"
    
    # 步骤5: 运行评估脚本
    echo -e "${YELLOW}[5/5] 运行评估脚本...${NC}"
    python "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/UCL-bench/local_evaluate_thinking.py" \
        --chatgpt_result_path "/caizhenyang/panghuaiwen/legal_LLM/dataset/result/res_result/qwen3_8B_eval_result.json" \
        --model_result_path "$MODEL_RESULT_PATH" \
        --datasource_path "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/UCL-bench/dataset/legal_data_sample.json" \
        --result_path "$SCORE_RESULT_PATH" \
        --model_path "/F00120250029/lixiang_share/Models/Qwen3-8B"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 评估脚本执行失败${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}评估流程完成！${NC}"
    echo -e "${GREEN}推理结果: $MODEL_RESULT_PATH${NC}"
    echo -e "${GREEN}评分结果: $SCORE_RESULT_PATH${NC}"
    echo -e "${GREEN}============================================${NC}"
}

# 运行主函数
main "$@"