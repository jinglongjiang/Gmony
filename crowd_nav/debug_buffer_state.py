#!/usr/bin/env python3
"""Debug当前buffer状态 - 需要在训练代码中调用"""

def debug_buffer_il_status(buffer, sample_count=10):
    """检查buffer中IL数据的实际状态"""
    import logging

    total = len(buffer.episodes)

    # 统计source字段
    il_count = 0
    rl_count = 0
    missing_count = 0
    other_count = 0

    for ep in buffer.episodes:
        meta = ep.get('meta', {})
        source = meta.get('source', None)

        if source == 'IL':
            il_count += 1
        elif source == 'RL':
            rl_count += 1
        elif source is None:
            missing_count += 1
        else:
            other_count += 1

    logging.warning(f"[BUFFER-DEBUG] Total episodes: {total}")
    logging.warning(f"[BUFFER-DEBUG] IL episodes: {il_count} ({il_count/total*100:.1f}%)")
    logging.warning(f"[BUFFER-DEBUG] RL episodes: {rl_count} ({rl_count/total*100:.1f}%)")
    if missing_count > 0:
        logging.warning(f"[BUFFER-DEBUG] Missing source: {missing_count}")
    if other_count > 0:
        logging.warning(f"[BUFFER-DEBUG] Other source: {other_count}")

    # 采样前几个episodes的meta
    logging.warning(f"[BUFFER-DEBUG] 前{sample_count}个episodes的meta['source']:")
    for i in range(min(sample_count, total)):
        meta = buffer.episodes[i].get('meta', {})
        source = meta.get('source', '<NONE>')
        event = meta.get('event', '<NONE>')
        logging.warning(f"  [{i}] source={source}, event={event}")

    # 测试PER采样逻辑
    il_pool = [i for i, ep in enumerate(buffer.episodes)
               if ep.get('meta', {}).get('source', 'RL') == 'IL']
    rl_pool = [i for i, ep in enumerate(buffer.episodes)
               if ep.get('meta', {}).get('source', 'RL') != 'IL']

    logging.warning(f"[BUFFER-DEBUG] PER采样pool: IL={len(il_pool)}, RL={len(rl_pool)}")

    if len(il_pool) == 0 and il_count > 0:
        logging.error(f"[BUFFER-DEBUG] ❌ BUG DETECTED: {il_count} episodes have source='IL' but il_pool is empty!")
        logging.error(f"[BUFFER-DEBUG] Checking condition logic:")
        # 检查第一个IL episode
        for i, ep in enumerate(buffer.episodes):
            if ep.get('meta', {}).get('source') == 'IL':
                meta = ep.get('meta', {})
                source_val = meta.get('source', 'RL')
                logging.error(f"  Found IL episode at index {i}")
                logging.error(f"    meta.get('source', 'RL') = '{source_val}'")
                logging.error(f"    Condition check: '{source_val}' == 'IL' -> {source_val == 'IL'}")
                break

if __name__ == '__main__':
    print("这个脚本需要在训练代码中import使用")
    print("在train.py中添加：")
    print("  from debug_buffer_state import debug_buffer_il_status")
    print("  debug_buffer_il_status(rl_buf)")
