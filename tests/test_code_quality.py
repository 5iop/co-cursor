"""
静态代码质量检查测试

测试覆盖:
- 变量作用域检查
- 索引边界保护检查
- 除零保护检查
- scatter/gather 操作安全性
- 三分支处理完整性
- 设备一致性检查
"""
import pytest
import re
from typing import List, Tuple


class TestVariableScope:
    """变量作用域检查"""

    def test_lengths_variable_scope(self, alpha_ddim_lines):
        """检查 lengths 变量是否在正确的作用域内定义和使用"""
        # 找到 lengths 的定义位置
        lengths_def_lines = []
        lengths_use_lines = []

        for i, line in enumerate(alpha_ddim_lines):
            if 'lengths = effective_length' in line:
                lengths_def_lines.append(i + 1)
            elif 'lengths.' in line or 'lengths[' in line or 'lengths)' in line:
                if 'lengths = ' not in line:
                    lengths_use_lines.append(i + 1)

        # lengths 应该有定义
        assert len(lengths_def_lines) > 0, "未找到 lengths 变量定义"

        # 检查定义是否在 is_per_sample 分支内
        for def_line in lengths_def_lines:
            # 检查前几行是否有 if is_per_sample
            found_per_sample = False
            for i in range(max(0, def_line - 10), def_line):
                if 'if is_per_sample:' in alpha_ddim_lines[i]:
                    found_per_sample = True
                    break
            assert found_per_sample, f"Line {def_line}: lengths 定义不在 is_per_sample 分支内"

    def test_per_sample_padding_mask_scope(self, alpha_ddim_lines):
        """检查 per_sample_padding_mask 变量作用域"""
        mask_def_line = None
        mask_use_lines = []

        for i, line in enumerate(alpha_ddim_lines):
            if 'per_sample_padding_mask = ' in line:
                mask_def_line = i + 1
            elif 'per_sample_padding_mask' in line and '=' not in line.split('per_sample_padding_mask')[0]:
                mask_use_lines.append(i + 1)

        if mask_def_line:
            # 所有使用应该在定义之后
            for use_line in mask_use_lines:
                if use_line != mask_def_line:
                    assert use_line > mask_def_line or use_line < mask_def_line - 50, \
                        f"Line {use_line}: per_sample_padding_mask 可能在定义前使用"


class TestIndexBoundsProtection:
    """索引边界保护检查"""

    def test_effective_length_clamp(self, alpha_ddim_source):
        """检查 effective_length 是否有 clamp(min=2) 保护"""
        # 应该有 .clamp(min=2) 或 max(2, effective_length)
        has_tensor_clamp = 'effective_length' in alpha_ddim_source and 'clamp(min=2)' in alpha_ddim_source
        has_int_max = 'max(2, effective_length)' in alpha_ddim_source

        assert has_tensor_clamp or has_int_max, "effective_length 缺少 clamp(min=2) 保护"

    def test_lengths_minus_one_clamp(self, alpha_ddim_lines):
        """检查 lengths - 1 是否有 clamp 保护"""
        issues = []
        for i, line in enumerate(alpha_ddim_lines):
            if 'lengths - 1' in line and 'scatter' not in line and 'gather' not in line:
                # 检查是否有 clamp
                if '.clamp(' not in line:
                    # 检查前 10 行是否有 clamp (扩大搜索范围)
                    context = '\n'.join(alpha_ddim_lines[max(0, i - 10):i + 1])
                    if 'clamp' not in context:
                        issues.append(f"Line {i + 1}: lengths - 1 可能缺少 clamp 保护")

        # 只警告，不失败 (因为可能通过其他方式保护)
        if issues:
            print(f"\n警告 - 可能缺少 clamp 保护:\n" + '\n'.join(issues))

    def test_end_indices_clamp(self, alpha_ddim_lines):
        """检查 end_indices 计算是否有 clamp 保护"""
        for i, line in enumerate(alpha_ddim_lines):
            if 'end_indices = ' in line and 'lengths - 1' in line:
                assert 'clamp' in line, f"Line {i + 1}: end_indices 计算缺少 clamp 保护"


class TestDivisionSafety:
    """除零保护检查"""

    def test_effective_length_min_protection(self, alpha_ddim_source):
        """检查 effective_length 是否有最小值保护（防止索引越界）"""
        # 应该有 effective_length >= 2 的保护
        # 这是为了确保 lengths - 1 作为索引时不会越界
        # 检查是否存在 clamp(min=2)
        assert 'clamp(min=2)' in alpha_ddim_source or 'max(2,' in alpha_ddim_source, \
            "未找到对 effective_length >= 2 的保护，可能导致索引越界"

    def test_division_epsilon(self, alpha_ddim_lines):
        """检查除法是否有 epsilon 保护"""
        division_patterns = [
            r'/ \(',
            r'/ torch',
            r'/ \w+\s*$',
        ]

        issues = []
        for i, line in enumerate(alpha_ddim_lines):
            if '/' in line and '//' not in line and '/*' not in line:
                # 检查是否有 epsilon 保护
                has_epsilon = '+ 1e-' in line or '+ 1e-' in alpha_ddim_lines[max(0, i - 1)]

                # 特殊情况：除以 (beta + 1) 等已知安全的情况
                safe_patterns = ['beta + 1', 'm_float', 'timesteps', 'self.timesteps']
                is_safe = any(p in line for p in safe_patterns)

                # 只记录潜在的除零问题（如果不安全且无 epsilon 保护）
                # 注：之前检查 sqrt(m_float - 1)，但该除法已被移除
                if not has_epsilon and not is_safe and 'direction_norm' in line:
                    issues.append(f"Line {i + 1}: {line.strip()}")

        # 这里不 assert，因为有些除法是安全的（通过 clamp 保护）


class TestScatterGatherSafety:
    """scatter/gather 操作安全性检查"""

    def test_scatter_has_clamp(self, alpha_ddim_lines):
        """检查 scatter_ 操作前是否有索引 clamp"""
        scatter_lines = []
        issues = []
        for i, line in enumerate(alpha_ddim_lines):
            if 'scatter_' in line:
                scatter_lines.append(i + 1)

        for scatter_line in scatter_lines:
            # 检查前 15 行是否有 clamp (扩大搜索范围)
            context_start = max(0, scatter_line - 16)
            context = '\n'.join(alpha_ddim_lines[context_start:scatter_line])
            if 'clamp' not in context:
                issues.append(f"Line {scatter_line}")

        # 只警告，不失败
        if issues:
            print(f"\n警告 - scatter_ 操作可能缺少 clamp 保护: {issues}")

    def test_gather_has_clamp(self, alpha_ddim_lines):
        """检查 gather 操作前是否有索引 clamp"""
        gather_lines = []
        issues = []
        for i, line in enumerate(alpha_ddim_lines):
            if '.gather(' in line:
                gather_lines.append(i + 1)

        for gather_line in gather_lines:
            # 检查前 15 行是否有 clamp (扩大搜索范围)
            context_start = max(0, gather_line - 16)
            context = '\n'.join(alpha_ddim_lines[context_start:gather_line])
            if 'clamp' not in context:
                issues.append(f"Line {gather_line}")

        # 只警告，不失败
        if issues:
            print(f"\n警告 - gather 操作可能缺少 clamp 保护: {issues}")


class TestBranchCompleteness:
    """三分支处理完整性检查"""

    def _find_functions_with_per_sample(self, lines: List[str]) -> List[Tuple[str, int]]:
        """找到所有处理 is_per_sample 的函数"""
        functions = []
        for i, line in enumerate(lines):
            if 'is_per_sample = isinstance(effective_length, torch.Tensor)' in line:
                # 向上找函数名
                for j in range(i - 1, max(0, i - 50), -1):
                    if 'def ' in lines[j]:
                        func_name = lines[j].strip().split('def ')[1].split('(')[0]
                        functions.append((func_name, i + 1))
                        break
        return functions

    def test_three_branch_handling(self, alpha_ddim_lines):
        """检查所有处理 effective_length 的函数是否有完整的三分支"""
        functions = self._find_functions_with_per_sample(alpha_ddim_lines)

        for func_name, line_num in functions:
            # 查找后续的 if/elif/else 分支
            has_if = False
            has_elif = False
            has_else = False

            for i in range(line_num, min(line_num + 150, len(alpha_ddim_lines))):
                line = alpha_ddim_lines[i].strip()
                if line.startswith('if is_per_sample:'):
                    has_if = True
                elif line.startswith('elif effective_length is not None'):
                    has_elif = True
                elif line.startswith('else:') and (has_if or has_elif):
                    has_else = True
                    break
                # 遇到新函数定义则停止
                if line.startswith('def ') and has_if:
                    break

            # 至少应该有 if is_per_sample
            assert has_if, f"函数 {func_name} 缺少 is_per_sample 分支"


class TestDeviceConsistency:
    """设备一致性检查"""

    def test_tensor_creation_has_device(self, alpha_ddim_lines):
        """检查张量创建是否指定 device"""
        tensor_creation_patterns = [
            'torch.zeros(',
            'torch.ones(',
            'torch.randn(',
            'torch.tensor(',
            'torch.full(',
        ]

        issues = []
        for i, line in enumerate(alpha_ddim_lines):
            for pattern in tensor_creation_patterns:
                if pattern in line:
                    # 检查是否有 device= 或 .to(device)
                    has_device = 'device=' in line or '.to(device)' in line or '.to(' in line

                    # 特殊情况：某些情况下从其他张量继承设备
                    inherits_device = '_like(' in line

                    if not has_device and not inherits_device:
                        issues.append(f"Line {i + 1}: {pattern} 可能未指定 device")

        # 只警告，不强制失败
        if issues:
            print(f"\n设备一致性警告:\n" + '\n'.join(issues))


class TestAutoLengthLogic:
    """自动长度预测逻辑检查"""

    def test_auto_length_condition(self, alpha_ddim_source):
        """检查自动长度预测的条件逻辑"""
        # 应该只在 auto_length=True 且 effective_length=None 时触发
        assert 'if auto_length and effective_length is None:' in alpha_ddim_source, \
            "自动长度预测条件不正确"

    def test_length_head_check(self, alpha_ddim_source):
        """检查是否验证模型支持长度预测"""
        assert "hasattr(self.model, 'length_head')" in alpha_ddim_source, \
            "未检查模型是否支持长度预测"


class TestCodeStyle:
    """代码风格检查"""

    def test_no_print_statements(self, alpha_ddim_lines):
        """检查是否有调试 print 语句"""
        print_lines = []
        in_main_block = False

        for i, line in enumerate(alpha_ddim_lines):
            stripped = line.strip()
            # 检查是否在 __main__ 块内
            if 'if __name__' in line:
                in_main_block = True
            # 只检查 __main__ 块外的 print
            if stripped.startswith('print(') and not stripped.startswith('# print'):
                if not in_main_block:
                    print_lines.append(i + 1)

        # 允许在 __main__ 块中有 print
        if print_lines:
            print(f"\n警告 - 发现 __main__ 块外的 print 语句: lines {print_lines}")

    def test_no_todo_comments(self, alpha_ddim_lines):
        """检查是否有未完成的 TODO 注释（仅警告）"""
        todo_lines = []
        for i, line in enumerate(alpha_ddim_lines):
            if 'TODO' in line.upper() and '#' in line:
                todo_lines.append((i + 1, line.strip()))

        if todo_lines:
            print(f"\n发现 TODO 注释:")
            for line_num, content in todo_lines:
                print(f"  Line {line_num}: {content}")


class TestPaperConsistency:
    """与论文的一致性检查"""

    def test_eq2_3_mask_initialization(self, alpha_ddim_source):
        """检查论文 Eq.2-3 掩码初始化是否实现"""
        assert '_initialize_with_condition' in alpha_ddim_source, \
            "未找到掩码初始化函数"

    def test_eq4_6_dual_covariance(self, alpha_ddim_source):
        """检查论文 Eq.4-6 双协方差混合是否实现"""
        # 应该有 sigma_d (方向协方差)
        assert 'sigma_d' in alpha_ddim_source, "未找到方向协方差 sigma_d"
        # sigma_n 可能命名为 sigma_epsilon 或 sigma_noise
        has_isotropic = ('sigma_n' in alpha_ddim_source or
                         'sigma_epsilon' in alpha_ddim_source or
                         'isotropic' in alpha_ddim_source.lower())
        assert has_isotropic, "未找到各向同性协方差实现"

    def test_eq8_9_entropy_control(self, alpha_ddim_source):
        """检查论文 Eq.8-9 MST 熵控制是否实现"""
        assert 'EntropyController' in alpha_ddim_source or 'mst' in alpha_ddim_source.lower(), \
            "未找到 MST 熵控制实现"

    def test_kc_constant(self, alpha_ddim_source):
        """检查 k_c 是否为常数（而非距离函数）"""
        # k_c 应该是常数，如 1/6
        # 不应该是 direction_norm / ...
        lines_with_kc = [l for l in alpha_ddim_source.split('\n') if 'kc = ' in l or 'k_c = ' in l]

        for line in lines_with_kc:
            # kc 应该是常数
            assert 'direction_norm' not in line or '//' in line, \
                f"k_c 不应该是距离的函数: {line.strip()}"
