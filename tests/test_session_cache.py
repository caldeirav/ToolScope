import toolscope
from toolscope.core.session import StickySessionConfig


def test_session_reuse_mode_respects_tag_filters(openai_tools, embedder):
    cfg = StickySessionConfig(
        enabled=True,
        similarity_threshold_reuse=0.0,   # force reuse whenever state exists
        similarity_threshold_refresh=0.0,
        sticky_keep=0,
    )
    idx = toolscope.index(
        openai_tools,
        embedder=embedder,
        normalizer=toolscope.AutoToolNormalizer(),
        session_cfg=cfg,
    )

    sid = "s1"

    # First call: selects some tools and stores session state
    tools1 = idx.filter("delete production", k=3, session_id=sid)
    assert tools1

    # Second call: should reuse, BUT deny dangerous => tool must be excluded even in reuse
    tools2 = idx.filter("delete production", k=3, session_id=sid, deny_tags=["dangerous"])
    names2 = [t["function"]["name"] for t in tools2]
    assert "dangerous_delete_prod" not in names2
