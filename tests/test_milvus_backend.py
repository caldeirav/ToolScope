import pytest
import toolscope


pytestmark = [
    pytest.mark.filterwarnings("ignore:pkg_resources is deprecated as an API\\.:DeprecationWarning"),
    pytest.mark.filterwarnings(
        "ignore:Deprecated call to `pkg_resources\\.declare_namespace\\('sphinxcontrib'\\)`\\.:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:Deprecated call to `pkg_resources\\.declare_namespace\\('zope'\\)`\\.:DeprecationWarning"
    ),
]

pymilvus = pytest.importorskip("pymilvus")


def test_milvus_backend_roundtrip(openai_tools, embedder, tmp_path):
    db = str(tmp_path / "toolscope_milvus.db")
    backend = toolscope.MilvusLiteBackend(uri=db)

    idx = toolscope.index(
        openai_tools,
        embedder=embedder,
        backend=backend,
        normalizer=toolscope.AutoToolNormalizer(),
        namespace="ns1",
    )

    tools = idx.filter("jira issue", k=2)
    assert tools
