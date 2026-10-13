#!/usr/bin/env python3
"""测试buffer的IL采样逻辑"""

def test_il_pool_logic():
    """模拟PER采样中il_pool的构建逻辑"""

    # 模拟dataset的meta（没有source字段）
    episodes_data = [
        {'meta': {'event': 'reaching_goal', 'flag_success': 1}},
        {'meta': {'event': 'collision', 'flag_collision': 1}},
        {'meta': {'source': 'IL', 'event': 'reaching_goal'}},
        {'meta': {'source': 'RL', 'event': 'collision'}},
    ]

    print("测试1：原始dataset meta（无source字段）")
    for i, ep in enumerate(episodes_data[:2]):
        meta = ep.get('meta', {})
        source_with_default = meta.get('source', 'RL')
        result = (source_with_default == 'IL')
        print(f"  [{i}] meta={meta}")
        print(f"       get('source', 'RL') = '{source_with_default}' == 'IL'? {result}")

    print("\n测试2：setdefault后（应该添加source='IL'）")
    for i, ep in enumerate(episodes_data[:2]):
        meta = ep.get('meta', {})
        meta.setdefault('source', 'IL')  # 模拟_meta_from_event
        print(f"  [{i}] meta after setdefault = {meta}")
        source_with_default = meta.get('source', 'RL')
        result = (source_with_default == 'IL')
        print(f"       get('source', 'RL') = '{source_with_default}' == 'IL'? {result}")

    print("\n测试3：显式设置的source字段")
    for i, ep in enumerate(episodes_data[2:]):
        meta = ep.get('meta', {})
        source_with_default = meta.get('source', 'RL')
        result = (source_with_default == 'IL')
        print(f"  [{i+2}] meta={meta}")
        print(f"       get('source', 'RL') = '{source_with_default}' == 'IL'? {result}")

    print("\n测试4：il_pool构建逻辑")
    test_episodes = [
        {'meta': {'source': 'IL', 'event': 'success'}},
        {'meta': {'source': 'RL', 'event': 'collision'}},
        {'meta': {'source': 'IL', 'event': 'success'}},
    ]

    il_pool = [i for i, ep in enumerate(test_episodes)
               if ep.get('meta', {}).get('source', 'RL') == 'IL']
    rl_pool = [i for i, ep in enumerate(test_episodes)
               if ep.get('meta', {}).get('source', 'RL') != 'IL']

    print(f"  Episodes: {len(test_episodes)}")
    print(f"  IL pool: {il_pool} (size={len(il_pool)})")
    print(f"  RL pool: {rl_pool} (size={len(rl_pool)})")

    # 统计actual IL count
    il_count_actual = sum(1 for ep in test_episodes if ep.get('meta', {}).get('source') == 'IL')
    print(f"  Actual IL episodes (source=='IL'): {il_count_actual}")

if __name__ == '__main__':
    test_il_pool_logic()
