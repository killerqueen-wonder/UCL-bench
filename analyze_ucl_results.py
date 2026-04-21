import json
import argparse

def analyze_results(score_path, inference_path):
    try:
        with open(score_path, 'r', encoding='utf-8') as f:
            score_data = json.load(f)
        
        with open(inference_path, 'r', encoding='utf-8') as f:
            inference_data = json.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    print("\n" + "="*85)
    print(" 📊 UCL-Bench 深度性能与成本评测报告 (非流式精简版)")
    print("="*85)

    # 全局统计变量
    g_base_score = 0
    g_local_score = 0
    g_items = 0
    
    g_total_time = 0.0
    g_tool_latency = 0.0
    g_rag_count = 0
    
    g_total_tokens = 0
    g_user_tokens = 0
    g_inter_tokens = 0
    g_comp_tokens = 0

    if not isinstance(score_data, dict):
        print("错误: 预期的 JSON 结构为按任务分类的字典。")
        return

    # 遍历每个任务
    for task_name, score_items in score_data.items():
        item_count = len(score_items)
        if item_count == 0: continue

        # 任务内统计变量
        t_base = 0; t_local = 0
        t_time = 0.0; t_tool = 0.0; t_rag = 0
        t_total_tok = 0; t_user = 0; t_inter = 0; t_comp = 0

        inf_task_items = inference_data.get(task_name, [])
        inf_lookup = {str(item.get("id")): item for item in inf_task_items}

        for s_item in score_items:
            # 得分统计
            scores = s_item.get("evaluation_score", [0, 0])
            t_base += scores[0] if len(scores) > 0 else 0
            t_local += scores[1] if len(scores) > 1 else 0

            # 性能与 Token 读取
            item_id = str(s_item.get("id"))
            inf_item = inf_lookup.get(item_id, {})
            
            t_time += inf_item.get("total_time_sec", 0.0)
            t_tool += inf_item.get("tool_latency_sec", 0.0)
            t_rag += inf_item.get("rag_count", 0)
            
            t_total_tok += inf_item.get("total_tokens", 0)
            t_user += inf_item.get("user_prompt_tokens", 0)
            t_inter += inf_item.get("inter_agent_tokens", 0)
            t_comp += inf_item.get("completion_tokens", 0)

        # 累加到全局
        g_base_score += t_base; g_local_score += t_local; g_items += item_count
        g_total_time += t_time; g_tool_latency += t_tool; g_rag_count += t_rag
        g_total_tokens += t_total_tok; g_user_tokens += t_user; g_inter_tokens += t_inter; g_comp_tokens += t_comp

        # 打印单任务统计
        print(f"【任务】: {task_name} (样本数: {item_count})")
        print(f"  [质量] 得分比例(Local/Base): {(t_local/t_base if t_base>0 else 0):.4f}  ({t_local}/{t_base})")
        print(f"  [耗时] 均次总时长: {t_time/item_count:.2f}s | 检索耗时: {t_tool/item_count:.2f}s | 检索频次: {t_rag/item_count:.1f}次")
        print(f"  [Token] 均次总计: {int(t_total_tok/item_count)} (User: {int(t_user/item_count)} | Inter: {int(t_inter/item_count)} | Comp: {int(t_comp/item_count)})")
        print("-" * 85)

    # 打印全局汇总
    print("\n🏆 【全局指标总览】")
    print(f"  ➤ 总评测样本数: {g_items}")
    print(f"  ➤ 全局得分比例 (Local/Base): {(g_local_score/g_base_score if g_base_score>0 else 0):.4f} ({g_local_score}/{g_base_score})")
    
    print("\n⏳ 【时间与交互延迟】 (平均每条 Query)")
    print(f"  - 整体端到端时长 (Total Time): {g_total_time/g_items:.2f} 秒")
    print(f"  - 外部检索阻塞时长 (Tool Latency): {g_tool_latency/g_items:.2f} 秒")
    print(f"  - 检索动作发起次数 (RAG Rounds): {g_rag_count/g_items:.2f} 次 (系统总计物理检索 {g_rag_count} 次)")
    
    print("\n💰 【Token 消耗清单】 (平均每条 Query)")
    print(f"  - [全流程总额] Total Tokens: {int(g_total_tokens/g_items)} (包含主推及脱水模型全周期消耗)")
    print(f"  - [直接输入] User Prompt Token: {int(g_user_tokens/g_items)} (用户的原始提问)")
    print(f"  - [直接产出] Completion Token : {int(g_comp_tokens/g_items)} (最终展示给用户的答案字数)")
    print(f"  - [系统损耗] Inter-Agent Token: {int(g_inter_tokens/g_items)} (模型试错、RAG召回上下文、内部交流)")
    
    # 计算损耗比
    print(f"\n💡 【系统评级】 内部损耗比 (Inter-Agent / Total Tokens): {(g_inter_tokens/g_total_tokens if g_total_tokens>0 else 0)*100:.1f}%")
    print("="*85 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析UCL非流式架构下的各项指标')
    parser.add_argument('--score_path', type=str, required=True, help='打分结果 JSON 路径')
    parser.add_argument('--inference_path', type=str, required=True, help='推理阶段生成的 JSON 路径(含Token统计)')
    
    args = parser.parse_args()
    analyze_results(args.score_path, args.inference_path)