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

# 显示帮助信息
show_help() {
    echo -e "${GREEN}用法: $0 [选项]${NC}"
    echo "选项:"
    echo "  -m, --test_model   测试模型名称 (如: RL2.0.3)"
    echo "  -d, --date         日期标识 (如: 0123)"
    echo "  -p, --model_path   完整模型路径 (必需)"
    echo "                     示例: /F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-2.0.3/global_step_40/actor_merge"
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

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    echo -e "${GREEN}============================================${NC}"
    echo -e "${GREEN}开始法律LLM评估流程${NC}"
    echo -e "${GREEN}模型标识: $TEST_MODEL${NC}"
    echo -e "${GREEN}日期标识: $DATE${NC}"
    echo -e "${GREEN}模型路径: $MODEL_PATH${NC}"
    echo -e "${GREEN}============================================${NC}"
    
    # 验证模型路径
    validate_model_path
    
    # 步骤3: 生成动态文件名
    echo -e "${YELLOW}[3/5] 生成输出文件名...${NC}"
    
    # 从模型路径中提取有用的信息用于文件名
    # 尝试从路径中提取更具体的模型信息
    if [[ "$MODEL_PATH" =~ legal_exam-ppo-qwen3-8b-RL-([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        MODEL_VERSION="${BASH_REMATCH[1]}"
        echo -e "${BLUE}从路径中提取到模型版本: $MODEL_VERSION${NC}"
        FILE_PREFIX="qwen3-8B_RL${MODEL_VERSION}"
    elif [[ "$MODEL_PATH" =~ legal_exam-ppo-qwen3-8b-([^/]+) ]]; then
        MODEL_TYPE="${BASH_REMATCH[1]}"
        echo -e "${BLUE}从路径中提取到模型类型: $MODEL_TYPE${NC}"
        FILE_PREFIX="qwen3-8B_${MODEL_TYPE}"
    else
        # 使用test_model参数
        FILE_PREFIX="qwen3-8B_${TEST_MODEL}"
    fi
    
    # 中间结果文件（推理结果）
    MODEL_RESULT_FILE="${FILE_PREFIX}_eval_result_${DATE}.json"
    MODEL_RESULT_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/res_result/${MODEL_RESULT_FILE}"
    
    # 最终评分结果文件
    SCORE_RESULT_FILE="${FILE_PREFIX}_score_${DATE}.json"
    SCORE_RESULT_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/score_result/${SCORE_RESULT_FILE}"
    
    echo -e "${BLUE}推理结果文件: $MODEL_RESULT_FILE${NC}"
    echo -e "${BLUE}评分结果文件: $SCORE_RESULT_FILE${NC}"
    
    # 步骤4: 运行推理脚本
    echo -e "${YELLOW}[4/5] 运行推理脚本...${NC}"
    echo -e "${BLUE}使用的模型路径: $MODEL_PATH${NC}"
    
    python "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/UCL-bench/local_inference_single_round.py" \
        --data_path "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/UCL-bench/dataset/legal_data_sample.json" \
        --model_path "$MODEL_PATH" \
        --result_path "$MODEL_RESULT_PATH" \
        --model_name "no_GPT" \
        --retriever true \
        --topk 10 \
        --max_turn 12 \
        --retrieve_path "http://127.0.0.1:8007/retrieve"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 推理脚本执行失败${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}推理完成，结果保存在: $MODEL_RESULT_PATH${NC}"
    
    # 步骤5: 运行评估脚本
    echo -e "${YELLOW}[5/5] 运行评估脚本...${NC}"
    python "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/UCL-bench/local_evaluate_thinking.py" \
        --chatgpt_result_path "/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/res_result/qwen3_8B_eval_result.json" \
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
    echo -e "${GREEN}模型标识: $TEST_MODEL${NC}"
    echo -e "${GREEN}模型路径: $MODEL_PATH${NC}"
    echo -e "${GREEN}推理结果: $MODEL_RESULT_PATH${NC}"
    echo -e "${GREEN}评分结果: $SCORE_RESULT_PATH${NC}"
    echo -e "${GREEN}============================================${NC}"
}

# 运行主函数
main "$@"