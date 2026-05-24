"""H&M-adapted Park-2023-lineage memory-retrieval-reflection-decision cognition pipeline.

Adapted from Fragment Labs' backend/simulation/cognition/ implementation
(originally tuned on WIP beverage data). All hyperparameters are frozen at
Fragment defaults; no tuning on H&M data. See ../../preregistration_v2.md.
"""

# Hyperparameter freeze (committed to pre-registration v2)
LLM_AFFECT_BLEND_RATIO = 0.6  # 60% LLM friction, 40% pre-computed affect
TOP_K_MEMORIES = 5
FRICTION_WEIGHTS = {
    "price": 0.20,
    "trust": 0.15,
    "decision": 0.20,
    "channel": 0.10,
    "memory": 0.15,
    "product_relevance": 0.20,
}
