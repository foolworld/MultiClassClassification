from dataset import load_dataset


def test_leaves_data():
    """测试树叶数据集加载"""
    print("=== 测试树叶数据集 ===")
    try:
        # 加载树叶数据
        (train_x, train_y), (val_x, val_y), classes = load_dataset('./data/dataset1')

        # 检查数据是否正确加载
        assert len(train_x) > 0, "训练集不能为空"
        assert len(val_x) > 0, "验证集不能为空"
        assert len(classes) == 8, f"树叶应该有8个类别，实际得到{len(classes)}个"

        print("✅ 树叶数据测试通过")
        print(f"   训练集: {train_x.shape}")
        print(f"   验证集: {val_x.shape}")
        print(f"   类别: {classes}")
        return True

    except Exception as e:
        print(f"❌ 树叶数据测试失败: {e}")
        return False


def test_butterflies_data():
    """测试蝴蝶数据集加载"""
    print("=== 测试蝴蝶数据集 ===")
    try:
        # 加载蝴蝶数据
        (train_x, train_y), (val_x, val_y), classes = load_dataset('./data/dataset2')

        # 检查数据是否正确加载
        assert len(train_x) > 0, "训练集不能为空"
        assert len(val_x) > 0, "验证集不能为空"
        assert len(classes) == 100, f"蝴蝶应该有100个类别，实际得到{len(classes)}个"

        print("✅ 蝴蝶数据测试通过")
        print(f"   训练集: {train_x.shape}")
        print(f"   验证集: {val_x.shape}")
        print(f"   类别数: {len(classes)}")
        return True

    except Exception as e:
        print(f"❌ 蝴蝶数据测试失败: {e}")
        return False


def test_data_format():
    """测试数据格式是否正确"""
    print("=== 测试数据格式 ===")
    try:
        # 测试其中一个数据集
        (train_x, train_y), (val_x, val_y), classes = load_dataset('./data/dataset1')

        # 检查数据格式
        assert train_x.dtype == np.float32, "图片数据类型应该是float32"
        assert np.max(train_x) <= 1.0, "图片数据应该归一化到0-1"
        assert np.min(train_x) >= 0.0, "图片数据应该归一化到0-1"
        assert train_x.shape[1] == 3, "图片应该是3通道(RGB)"
        assert train_x.shape[2] == 28, "图片高度应该是28"
        assert train_x.shape[3] == 28, "图片宽度应该是28"

        print("✅ 数据格式测试通过")
        return True

    except Exception as e:
        print(f"❌ 数据格式测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始数据加载测试...")
    print("=" * 50)

    # 运行所有测试
    test1 = test_leaves_data()
    print()
    test2 = test_butterflies_data()
    print()
    test3 = test_data_format()

    print("=" * 50)
    if test1 and test2 and test3:
        print("🎉 所有测试通过！数据加载器工作正常")
        print("💡 提示: 现在可以删除 test_dataloader.py 文件了")
    else:
