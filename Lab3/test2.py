import unittest
from graph import TextGraph  # 假设模块可正确导入

class TestFindBridgeWords(unittest.TestCase):
    def setUp(self):
        """初始化测试图，构建测试所需的边"""
        self.graph = TextGraph()
        # 构建图：apple -> banana -> cherry，apple -> date，banana -> date
        test_text = "apple banana cherry apple date banana"
        words = self.graph.process_text(test_text)
        self.graph.build_graph(words)

        # 测试用例1：节点不存在（覆盖基本路径1）
    def test_nodes_not_exist(self):
        word1 = "grape"  # 图中不存在 grape
        word2 = "cherry"
        expected_bridges = []
        actual_bridges = self.graph.find_bridge_words(word1, word2)
        self.assertEqual(actual_bridges, expected_bridges, "节点不存在时未返回空列表")

        # 测试用例2：无桥接词（覆盖基本路径2）
    def test_no_bridge_words(self):
        word1 = "banana"
        word2 = "cherry"  # apple -> date 直接相连，无中间桥接词
        expected_bridges = []
        actual_bridges = self.graph.find_bridge_words(word1, word2)
        self.assertEqual(actual_bridges, expected_bridges, "无桥接词时返回非空列表")

    # 测试用例3：存在桥接词（覆盖基本路径3）
    def test_bridge_words_exist(self):
        word1 = "apple"
        word2 = "cherry"
        expected_bridges = ["banana"]
        actual_bridges = self.graph.find_bridge_words(word1, word2)
        self.assertEqual(actual_bridges, expected_bridges, "桥接词存在时返回错误")
        
     # 测试用例4：输入空参数
    def test_any_parameter_empty(self):
        # 场景1：word1为空，word2为有效字符串
        actual1 = self.graph.find_bridge_words("", "cherry")
        # 场景2：word2为空，word1为有效字符串
        actual2 = self.graph.find_bridge_words("apple", "")
        # 场景3：word1和word2均为空
        actual3 = self.graph.find_bridge_words("", "")
        
        expected = []
        self.assertEqual(actual1, expected, "空word1未返回空列表")
        self.assertEqual(actual2, expected, "空word2未返回空列表")
        self.assertEqual(actual3, expected, "双空参数未返回空列表")


if __name__ == "__main__":
    unittest.main(verbosity=2)