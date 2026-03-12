#!/bin/bash
# CommonGEN-GAIL 快速启动脚本

set -e  # 遇到错误立即退出

echo "=============================================="
echo "  CommonGEN-GAIL 快速启动脚本"
echo "=============================================="
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 显示菜单
echo "请选择要执行的操作："
echo ""
echo "  1) 数据预处理 - 小规模测试（100条）"
echo "  2) 数据预处理 - 完整数据（5000条）"
echo "  3) 判别器独立测试（当前数据）"
echo "  4) 查看判别器容量信息"
echo "  5) 查看项目结构"
echo "  6) 清理日志和临时文件"
echo "  0) 退出"
echo ""
read -p "请输入选项 [0-6]: " choice

case $choice in
  1)
    echo -e "${GREEN}[1/3] 开始数据预处理（小规模）...${NC}"
    cd code
    python preprocess_commongenv2.py --train_size 100 --val_size 20 --version v2
    echo -e "${GREEN}✅ 数据预处理完成！${NC}"
    echo "输出位置: ../data/data_commongenv/"
    ;;
    
  2)
    echo -e "${YELLOW}警告: 处理5000条数据可能需要几分钟...${NC}"
    read -p "确认继续? [y/N]: " confirm
    if [[ $confirm == [yY] ]]; then
      echo -e "${GREEN}[1/3] 开始数据预处理（完整数据）...${NC}"
      cd code
      python preprocess_commongenv2.py --train_size 5000 --val_size 500 --version v2
      echo -e "${GREEN}✅ 数据预处理完成！${NC}"
      echo "输出位置: ../data/data_commongenv/"
    else
      echo "已取消"
    fi
    ;;
    
  3)
    echo -e "${GREEN}[2/3] 开始判别器测试...${NC}"
    LOG_FILE="../logs/test_$(date +%Y%m%d_%H%M%S).log"
    echo "日志将保存到: $LOG_FILE"
    cd code
    CUDA_VISIBLE_DEVICES=0 python test_discriminator_commongenv.py 2>&1 | tee "$LOG_FILE"
    echo -e "${GREEN}✅ 测试完成！${NC}"
    echo "查看日志: $LOG_FILE"
    ;;
    
  4)
    echo -e "${GREEN}[3/3] 查看判别器容量信息...${NC}"
    cd code
    python fix_discriminator_capacity.py
    ;;
    
  5)
    echo -e "${GREEN}项目结构:${NC}"
    echo ""
    echo "CommonGEN/"
    echo "├── code/          # 代码文件（3个脚本）"
    echo "├── data/          # 数据文件"
    echo "│   └── data_commongenv/  # CoT格式的轨迹数据"
    echo "├── checkpoints/   # 模型检查点（训练后）"
    echo "├── logs/          # 日志文件"
    echo "├── docs/          # 文档和计划"
    echo "└── README.md      # 项目说明"
    echo ""
    
    echo -e "${GREEN}文件统计:${NC}"
    echo "代码文件: $(ls -1 code/*.py 2>/dev/null | wc -l)"
    echo "数据文件: $(ls -1 data/data_commongenv/*.json 2>/dev/null | wc -l)"
    echo "日志文件: $(ls -1 logs/*.log 2>/dev/null | wc -l)"
    echo "文档文件: $(ls -1 docs/*.md 2>/dev/null | wc -l)"
    
    # 统计数据量
    if [ -f "data/data_commongenv/commongenv_train_trajectories.json" ]; then
      TRAIN_COUNT=$(python3 -c "import json; print(len(json.load(open('data/data_commongenv/commongenv_train_trajectories.json'))))" 2>/dev/null || echo "?")
      VAL_COUNT=$(python3 -c "import json; print(len(json.load(open('data/data_commongenv/commongenv_val_trajectories.json'))))" 2>/dev/null || echo "?")
      echo ""
      echo -e "${GREEN}数据量:${NC}"
      echo "训练集: $TRAIN_COUNT 条"
      echo "验证集: $VAL_COUNT 条"
    fi
    ;;
    
  6)
    echo -e "${YELLOW}准备清理日志和临时文件...${NC}"
    echo "将删除:"
    echo "  - logs/*.log"
    echo "  - checkpoints/*.pt"
    echo "  - __pycache__/"
    read -p "确认删除? [y/N]: " confirm
    if [[ $confirm == [yY] ]]; then
      rm -f logs/*.log
      rm -f checkpoints/*.pt
      find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
      echo -e "${GREEN}✅ 清理完成！${NC}"
    else
      echo "已取消"
    fi
    ;;
    
  0)
    echo "再见！"
    exit 0
    ;;
    
  *)
    echo -e "${RED}无效选项！${NC}"
    exit 1
    ;;
esac

echo ""
echo "=============================================="
echo -e "${GREEN}操作完成！${NC}"
echo "=============================================="
echo ""
echo "📖 查看文档: cat README.md"
echo "📖 实验计划: cat docs/COMMONGENV_EXPERIMENT_PLAN.md"
echo "📖 下一步: cat docs/NEXT_STEPS.md"
echo ""

