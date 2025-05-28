import unittest
from graph import TextGraph  # 假设模块路径正确

class TestGenerateNewText(unittest.TestCase):
    def setUp(self):
        """初始化测试图，构建不同场景的图结构"""
        self.graph = TextGraph()
        self.graph.build_graph(self.graph.process_text("apple banana cherry"))  # apple -> banana -> cherry
        self.graph_no_bridge = TextGraph()
        self.graph_no_bridge.build_graph(self.graph_no_bridge.process_text("banana date"))  # banana -> date

    # 有效等价类测试
    def test_valid_class_1_normal_case(self):
        """[有效类(1)] 正常文本含桥接词，应插入桥接词"""
        input_text = "Apple cherry"
        expected = "Apple Banana cherry"
        actual = self.graph.generate_new_text(input_text)
        self.assertEqual(actual, expected, f"预期：{expected}，实际：{actual}")

    def test_valid_class_2_punctuation_case(self):
        """[有效类(2)] 含标点文本，应过滤标点并插入桥接词"""
        input_text = "apple,cherry!?"
        expected = "apple banana cherry"
        actual = self.graph.generate_new_text(input_text)
        self.assertEqual(actual, expected, f"预期：{expected}，实际：{actual}")

    def test_valid_class_3_multi_word_no_bridge(self):
        """[有效类(3)] 多单词无桥接词，应保持原文本"""
        input_text = "Banana date"
        expected = "Banana date"
        actual = self.graph_no_bridge.generate_new_text(input_text)
        self.assertEqual(actual, expected, f"预期：{expected}，实际：{actual}")

    # 无效等价类测试（带异常和边界检查）
    def test_invalid_class_4_empty_string(self):
        """[无效类(4)] 空字符串，应返回空"""
        input_text = ""
        expected = ""
        actual = self.graph.generate_new_text(input_text)
        self.assertEqual(actual, expected, f"预期：{expected}，实际：{actual}")

    def test_invalid_class_5_non_string_input(self):
        """[无效类(5)] 非字符串输入，应抛出TypeError"""
        with self.assertRaises(TypeError) as context:
            self.graph.generate_new_text(123)  # 传入整数
        self.assertEqual(str(context.exception), "输入必须为字符串", "异常信息不匹配")

    def test_invalid_class_6_single_word(self):
        """[无效类(6)] 单单词输入，应保持原文本"""
        input_text = "Apple"
        expected = "Apple"
        actual = self.graph.generate_new_text(input_text)
        self.assertEqual(actual, expected, f"预期：{expected}，实际：{actual}")

if __name__ == "__main__":
    unittest.main(verbosity=2)  # 设置verbosity=2显示详细结果