#!/bin/bash
# 快速运行特征稳定性分析脚本

echo "========================================"
echo "特征稳定性分析 - 快速模式"
echo "========================================"
echo ""

# 1. 首先运行测试（单个癌症）
echo "🧪 步骤1: 运行测试..."
echo "   测试癌症: BLCA"
echo "   预计时间: 2-5分钟"
echo ""

read -p "按Enter开始测试运行..."

python run_stability_analysis_fast.py --mode test --cancer BLCA --xai shap

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 测试成功！"
    echo ""
    echo "📂 查看测试结果:"
    ls -lh ../stability_analysis_test/BLCA/shap/ | head -10
    echo ""
    
    read -p "测试通过，是否继续完整分析？(y/n): " answer
    
    if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
        echo ""
        echo "🚀 步骤2: 开始完整分析..."
        echo "   预计时间: 2-8 小时"
        echo "   建议使用后台运行"
        echo ""
        
        read -p "选择运行方式 (1=前台运行, 2=后台运行): " mode
        
        if [ "$mode" == "2" ]; then
            echo "启动后台运行..."
            nohup python run_stability_analysis_fast.py --mode full > stability_analysis.log 2>&1 &
            
            pid=$!
            echo ""
            echo "✅ 已启动后台进程，PID: $pid"
            echo ""
            echo "📊 查看实时进度:"
            echo "   tail -f stability_analysis.log"
            echo ""
            echo "🛑 停止运行:"
            echo "   kill $pid"
            echo ""
            echo "等待3秒后显示初始输出..."
            sleep 3
            tail -20 stability_analysis.log
        else
            echo "启动前台运行..."
            python run_stability_analysis_fast.py --mode full
        fi
    else
        echo "已取消完整分析"
    fi
else
    echo ""
    echo "❌ 测试失败，请检查错误信息"
    exit 1
fi
























