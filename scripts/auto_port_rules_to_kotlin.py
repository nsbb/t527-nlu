#!/usr/bin/env python3
"""apply_post_rules의 단순 패턴 규칙을 자동으로 Kotlin Regex 코드로 변환.

지원 패턴 (자동):
  if re.search(r'PATTERN', text):
      [if preds['fn'] in (...) and preds['param_direction'] in (...):]
          preds['fn'] = '...'
          preds['param_direction'] = '...'
          preds['exec_type'] = '...'

복잡 패턴 (수동 표시):
  - 중첩 if (3단계 이상)
  - re.search 결과에 group() 사용
  - 변수 할당이 있는 경우 (m = re.search(...))
"""
import re
import sys

SRC = 'scripts/ensemble_inference_with_rules.py'
DST = 'docs/auto_ported_rules.kt'

# 함수 본문 추출 (apply_post_rules 24줄~1895줄)
with open(SRC) as f:
    lines = f.readlines()

func_body = lines[24:1895]  # 0-indexed

# 단순 if 블록 패턴 매칭
# pattern_block: "    if re.search(r'PATTERN', text):"
# action_lines:  "        preds['xxx'] = 'yyy'"
# nested_if:     "        if preds['fn'] in (...):"

class Block:
    def __init__(self):
        self.id = ""
        self.version = ""
        self.comment = []
        self.regex = ""
        self.fn_check = []  # in (...) 조건
        self.dir_check = []
        self.actions = []  # [(field, value)]
        self.complex = False
        self.raw_lines = []

def parse_blocks(body):
    """블록별로 파싱. 단순한 if-pattern-action 구조만 자동 변환."""
    blocks = []
    i = 0
    while i < len(body):
        line = body[i]
        # version comment 또는 일반 코멘트
        m_comment = re.match(r'^\s*#\s*(?:(v\d+):\s*)?(.+)$', line)
        if m_comment and not line.strip().startswith('#'):
            i += 1
            continue
        
        # if re.search(r'...', text): 패턴 시작
        m = re.match(r'^\s*if\s+re\.search\(r[\'"]([^\'"]+)[\'"]?,\s*text\):\s*$', line)
        if m:
            block = Block()
            block.regex = m.group(1)
            # 이전 코멘트 라인 추출 (위 1~3줄)
            for j in range(max(0, i-3), i):
                cl = body[j].strip()
                if cl.startswith('#'):
                    block.comment.append(cl)
                    vm = re.search(r'v(\d+)', cl)
                    if vm and not block.version:
                        block.version = f'v{vm.group(1)}'
            
            block.raw_lines.append(line)
            i += 1
            
            # 다음 들여쓰기 라인들 처리 (8칸 들여쓰기)
            while i < len(body):
                inner = body[i]
                if not inner.strip() or (not inner.startswith('        ') and not inner.startswith('\t\t')):
                    if inner.strip().startswith('#') and not block.actions:
                        i += 1
                        continue
                    break
                block.raw_lines.append(inner)
                
                # nested if
                m_if = re.match(r'^\s+if\s+preds\[[\'"](\w+)[\'"]\]\s*(in|==|!=)\s*(.+):\s*$', inner)
                if m_if:
                    field, op, rhs = m_if.group(1), m_if.group(2), m_if.group(3).strip()
                    if op == 'in':
                        # tuple/list 추출
                        items = re.findall(r"['\"]([^'\"]+)['\"]", rhs)
                        if field == 'fn':
                            block.fn_check = items
                        elif field == 'param_direction':
                            block.dir_check = items
                    else:
                        block.complex = True
                    i += 1
                    continue
                
                # action: preds['xxx'] = 'yyy'
                m_set = re.match(r'^\s+preds\[[\'"](\w+)[\'"]\]\s*=\s*[\'"]([^\'"]+)[\'"]\s*$', inner)
                if m_set:
                    block.actions.append((m_set.group(1), m_set.group(2)))
                    i += 1
                    continue
                
                # 그 외 (m = re.search, value 추출, 변수 사용 등) → complex
                if 'preds[' in inner or 're.search' in inner or 'self.' in inner:
                    block.complex = True
                i += 1
            
            blocks.append(block)
            continue
        i += 1
    return blocks

blocks = parse_blocks(func_body)
print(f"총 블록: {len(blocks)}", file=sys.stderr)
print(f"  단순 (자동 변환 가능): {sum(1 for b in blocks if not b.complex and b.actions)}", file=sys.stderr)
print(f"  복잡 (수동 처리 필요): {sum(1 for b in blocks if b.complex)}", file=sys.stderr)
print(f"  action 없음 (스킵): {sum(1 for b in blocks if not b.actions and not b.complex)}", file=sys.stderr)

# Python regex → Kotlin regex 변환
def py_regex_to_kotlin(pattern):
    """대부분 호환되지만 일부 escape 처리"""
    return pattern.replace('\\', '\\\\')  # \s → \\s, \d → \\d 등

# Kotlin 코드 생성
def gen_kotlin(blocks):
    lines = []
    lines.append("// 자동 생성된 PostRules — apply_post_rules의 단순 규칙들")
    lines.append("// 출처: scripts/ensemble_inference_with_rules.py")
    lines.append("// 자동 변환: scripts/auto_port_rules_to_kotlin.py")
    lines.append("")
    
    auto_count = 0
    for b in blocks:
        if b.complex or not b.actions:
            continue
        
        kotlin_pat = py_regex_to_kotlin(b.regex)
        
        # 코멘트
        if b.comment:
            lines.append(f"// {' '.join(b.comment).replace('#', '').strip()}")
        elif b.version:
            lines.append(f"// {b.version}")
        
        # if regex match
        lines.append(f'if (Regex("{kotlin_pat}").containsMatchIn(text)) {{')
        
        # nested fn/dir check
        indent = "    "
        if b.fn_check or b.dir_check:
            conds = []
            if b.fn_check:
                fn_set = ", ".join(f'"{x}"' for x in b.fn_check)
                conds.append(f"p.fn in setOf({fn_set})")
            if b.dir_check:
                dir_set = ", ".join(f'"{x}"' for x in b.dir_check)
                conds.append(f"p.direction in setOf({dir_set})")
            cond_str = " && ".join(conds)
            lines.append(f"{indent}if ({cond_str}) {{")
            indent = "        "
        
        # actions
        for field, value in b.actions:
            kt_field = {'param_direction': 'direction', 'exec_type': 'execType', 'param_type': 'paramType'}.get(field, field)
            lines.append(f'{indent}p.{kt_field} = "{value}"')
        
        if b.fn_check or b.dir_check:
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")
        auto_count += 1
    
    print(f"✅ Kotlin 자동 변환 완료: {auto_count}개 규칙", file=sys.stderr)
    return "\n".join(lines)

kotlin_code = gen_kotlin(blocks)
with open(DST, 'w') as f:
    f.write(kotlin_code)

# 통계
print(f"\n=== 결과 ===", file=sys.stderr)
print(f"  출력: {DST}", file=sys.stderr)
print(f"  라인 수: {len(kotlin_code.split(chr(10)))}", file=sys.stderr)
