23/2/2026:
- Fix 3D Viewer issue:
    - Reduce latecy from ??s to ~0ms (resource cache)/~1s (browser HTTP cache)
neo4j init:
```
NEO4J_HOME=/tmp/neo4j-community-5.26.0
# Set initial password non-interactively
$NEO4J_HOME/bin/neo4j-admin dbms set-initial-password password 2>&1
# Start
$NEO4J_HOME/bin/neo4j start 2>&1
sleep 8
# Verify bolt port
ss -tlnp | grep 7687 || echo "Port not up yet"
```

<!-- #TODO: -->
- Explain query later: make sure no silent fallback and explain what is the template use for.


@17/03
- ifx padding for training eval

apply_chat_template 已经把 system/user/assistant 消息组装成完整的 ChatML 字符串了，包含所有 special tokens（<|im_start|>, <|im_end|> 等）。

当这个字符串传给 tokenizer 时：

add_special_tokens=False（训练用的）：tokenizer 只做 tokenize，不额外加 BOS/EOS token。这是正确的，因为 chat template 已经包含了所有需要的 special tokens。
add_special_tokens=True（eval 默认的）：tokenizer 会再加一次 BOS token，导致序列开头多了一个 token。模型训练时从没见过这种格式，所以输出质量大幅下降。
padding=True 对单条样本无影响（只有 batch>1 时才需要 pad），但为了一致性也去掉。

简单说：eval 的 tokenizer 多加了一个 BOS token，导致整个输入 token 序列和训练时不一样，模型就"懵"了。