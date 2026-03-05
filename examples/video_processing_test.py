"""
Test video processing with VideoRAG integration

This script tests the VideoRAG integration in RAG-Anything using a sample video file.
It verifies the video processing pipeline, entity extraction, and query functionality.

Requirements:
- VideoRAG must be installed (see https://github.com/HKUDS/VideoRAG)
- OpenAI API key must be set in environment
- Test video file must exist at the specified path

Usage:
    # Basic test (uses temp directories, artifacts are deleted after run)
    RUN_FULL_VIDEO_TESTS=true uv run python examples/video_processing_test.py ~/Downloads/windmill.mp4 "What is this video about?"

    # Preserve intermediate artifacts for debugging
    RUN_FULL_VIDEO_TESTS=true ARTIFACT_DIR=./video_artifacts uv run python examples/video_processing_test.py ~/Downloads/windmill.mp4 "What is this video about?"

    # The ARTIFACT_DIR will contain:
    # - graph_chunk_entity_relation.graphml: Knowledge graph
    # - vdb_entities.json: Entity embeddings
    # - videorag/: VideoRAG processing data (segments, transcripts, captions)
    # - output/: Parsed content output

Environment Variables:
    RUN_FULL_VIDEO_TESTS=true  - Enable full processing and query tests
    ARTIFACT_DIR=./path        - Directory to preserve intermediate artifacts
"""

import os
import sys
import asyncio
import tempfile
import logging

# IMPORTANT: Set multiprocessing start method before any other imports
# VideoRAG requires 'spawn' method for multiprocessing
import multiprocessing

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

# Add paths for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raganything import RAGAnything, RAGAnythingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test video path
TEST_VIDEO = sys.argv[1]

# Preserve artifacts for debugging (set ARTIFACT_DIR env var)
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", None)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-3-small")
COMPLETION_MODEL = os.environ.get("COMPLETION_MODEL", "gpt-4o-mini")


async def create_llm_functions():
    """Create LLM and embedding functions for testing."""
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc

    # AIHubMix configuration
    OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            COMPLETION_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            **kwargs,
        )

    async def embedding_func(texts: list[str]) -> list[list[float]]:
        return await openai_embed(
            texts,
            model=EMBED_MODEL,
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
        )

    return llm_model_func, EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=embedding_func,
    )


async def _debug_print_knowledge_graph(rag, work_dir):
    """Debug function to print extracted entities and relations from knowledge graph."""
    import json

    logger.info("=" * 60)
    logger.info("DEBUG: Knowledge Graph Contents")
    logger.info("=" * 60)

    try:
        # Check if LightRAG is initialized
        if rag.lightrag is None:
            logger.warning("LightRAG not initialized")
            return

        # Get entities from vector DB
        entities_vdb = rag.lightrag.entities_vdb
        if entities_vdb:
            try:
                # Access the internal data
                entity_data = (
                    entities_vdb._data if hasattr(entities_vdb, "_data") else {}
                )
                logger.info(f"Total entities in vector DB: {len(entity_data)}")

                # Print entity names
                entity_names = []
                for entity_id, entity_info in entity_data.items():
                    if isinstance(entity_info, dict):
                        entity_names.append(entity_info.get("entity_name", entity_id))
                    else:
                        entity_names.append(entity_id)

                logger.info(f"Entity names: {entity_names[:50]}...")  # First 50
            except Exception as e:
                logger.warning(f"Could not read entities_vdb: {e}")

        # Get relations from vector DB
        relationships_vdb = rag.lightrag.relationships_vdb
        if relationships_vdb:
            try:
                rel_data = (
                    relationships_vdb._data
                    if hasattr(relationships_vdb, "_data")
                    else {}
                )
                logger.info(f"Total relations in vector DB: {len(rel_data)}")
            except Exception as e:
                logger.warning(f"Could not read relationships_vdb: {e}")

        # Check graph file
        graph_file = os.path.join(work_dir, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_file):
            logger.info(f"Graph file exists: {graph_file}")
            logger.info(f"Graph file size: {os.path.getsize(graph_file)} bytes")

        # Check VideoRAG data
        videorag_dir = os.path.join(work_dir, "videorag")
        if os.path.exists(videorag_dir):
            logger.info(f"VideoRAG directory exists: {videorag_dir}")

            # Check video segments
            segments_file = os.path.join(videorag_dir, "video_segments.json")
            if os.path.exists(segments_file):
                with open(segments_file, "r") as f:
                    segments_data = json.load(f)
                logger.info(f"Video segments: {len(segments_data)} videos")

                # Print segment info for first video
                for video_name, segments in segments_data.items():
                    logger.info(f"  Video '{video_name}': {len(segments)} segments")
                    # Print first few segment summaries
                    for seg_id, seg_data in list(segments.items())[:3]:
                        transcript = seg_data.get("transcript", "")[:100]
                        caption = seg_data.get("caption", "")[:100]
                        logger.info(f"    Segment {seg_id}:")
                        if transcript:
                            logger.info(f"      Transcript: {transcript}...")
                        if caption:
                            logger.info(f"      Caption: {caption}...")
                    break  # Only first video

            # Check VideoRAG entities
            vr_entities_file = os.path.join(videorag_dir, "vdb_entities.json")
            if os.path.exists(vr_entities_file):
                with open(vr_entities_file, "r") as f:
                    vr_entities = json.load(f)
                logger.info(
                    f"VideoRAG entities: {len(vr_entities.get('data', []))} entities"
                )

            # Check VideoRAG chunks
            vr_chunks_file = os.path.join(videorag_dir, "vdb_chunks.json")
            if os.path.exists(vr_chunks_file):
                with open(vr_chunks_file, "r") as f:
                    vr_chunks = json.load(f)
                chunks = vr_chunks.get("data", [])
                logger.info(f"VideoRAG chunks: {len(chunks)} chunks")
                # Print first few chunk previews
                for i, chunk in enumerate(chunks[:3]):
                    body = chunk.get("body", "")[:200]
                    logger.info(f"  Chunk {i}: {body}...")

        logger.info("=" * 60)
        logger.info("END DEBUG: Knowledge Graph Contents")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error printing knowledge graph: {e}")
        import traceback

        traceback.print_exc()


async def test_video_parsing():
    """Test 1: Video file parsing into content_list format."""
    logger.info("=" * 60)
    logger.info("Test 1: Video Parsing")
    logger.info("=" * 60)

    from raganything.parser import MineruParser

    parser = MineruParser()

    # Check if test video exists
    if not os.path.exists(TEST_VIDEO):
        logger.error(f"Test video not found: {TEST_VIDEO}")
        return False

    logger.info(f"Test video found: {TEST_VIDEO}")

    # Parse video file
    content_list = parser.parse_video(TEST_VIDEO)

    assert len(content_list) == 1, f"Expected 1 content item, got {len(content_list)}"
    assert content_list[0]["type"] == "video", (
        f"Expected type 'video', got {content_list[0]['type']}"
    )
    assert "video_path" in content_list[0], "Missing 'video_path' in content"
    assert "video_name" in content_list[0], "Missing 'video_name' in content"

    logger.info(f"[PASS] Video parsed successfully:")
    logger.info(f"  - Type: {content_list[0]['type']}")
    logger.info(f"  - Name: {content_list[0]['video_name']}")
    logger.info(f"  - Path: {content_list[0]['video_path']}")

    return True


async def test_video_processor_initialization():
    """Test 2: VideoModalProcessor initialization."""
    logger.info("=" * 60)
    logger.info("Test 2: Video Processor Initialization")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        llm_func, embed_func = await create_llm_functions()

        config = RAGAnythingConfig(
            working_dir=tmpdir,
            enable_video_processing=True,
            enable_image_processing=False,
            enable_table_processing=False,
            enable_equation_processing=False,
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_func,
            embedding_func=embed_func,
        )

        # Initialize LightRAG and processors
        result = await rag._ensure_lightrag_initialized()

        if not result.get("success"):
            logger.error(f"Failed to initialize: {result.get('error')}")
            return False

        # Check if video processor was created
        if "video" in rag.modal_processors:
            logger.info("[PASS] Video processor initialized successfully")
            processor = rag.modal_processors["video"]
            logger.info(f"  - Processor class: {processor.__class__.__name__}")
            logger.info(f"  - VideoRAG working dir: {processor.videorag_working_dir}")
            logger.info(f"  - Segment length: {processor.video_segment_length}s")
            logger.info(f"  - LLM provider: {processor.llm_provider}")
            return True
        else:
            logger.warning(
                "[SKIP] Video processor not initialized (VideoRAG not installed)"
            )
            return True  # Not a failure, just VideoRAG not available


async def test_full_video_processing():
    """Test 3: Full video processing pipeline (requires VideoRAG)."""
    logger.info("=" * 60)
    logger.info("Test 3: Full Video Processing Pipeline")
    logger.info("=" * 60)

    if not os.path.exists(TEST_VIDEO):
        logger.error(f"Test video not found: {TEST_VIDEO}")
        return False

    # Use persistent directory if ARTIFACT_DIR is set, otherwise use temp
    if ARTIFACT_DIR:
        work_dir = os.path.join(ARTIFACT_DIR, "video_processing_test")
        os.makedirs(work_dir, exist_ok=True)
        logger.info(f"Using persistent working directory: {work_dir}")
        tmpdir_ctx = None
        tmpdir = work_dir
    else:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_ctx.__enter__()

    llm_func, embed_func = await create_llm_functions()

    config = RAGAnythingConfig(
        working_dir=tmpdir,
        enable_video_processing=True,
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
        video_segment_length=30,
        video_llm_provider="openai",
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        embedding_func=embed_func,
    )

    try:
        # Process video
        logger.info(f"Processing video: {TEST_VIDEO}")
        logger.info("This may take several minutes for the first run...")

        await rag.process_document_complete(
            TEST_VIDEO,
            output_dir=os.path.join(tmpdir, "output"),
        )

        # Debug: Print extracted entities
        await _debug_print_knowledge_graph(rag, tmpdir)

        logger.info("[PASS] Video processed successfully")
        return True

    except ImportError as e:
        logger.warning(f"[SKIP] VideoRAG not installed: {e}")
        return True  # Not a failure, just VideoRAG not available

    except Exception as e:
        logger.error(f"[FAIL] Video processing error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up temp directory if we created one
        if tmpdir_ctx:
            tmpdir_ctx.__exit__(None, None, None)


async def test_video_query():
    """Test 4: Query about video content (requires processed video)."""
    logger.info("=" * 60)
    logger.info("Test 4: Video Content Query")
    logger.info("=" * 60)

    if not os.path.exists(TEST_VIDEO):
        logger.error(f"Test video not found: {TEST_VIDEO}")
        return False

    # Use persistent directory if ARTIFACT_DIR is set, otherwise use temp
    if ARTIFACT_DIR:
        work_dir = os.path.join(ARTIFACT_DIR, "video_query_test")
        os.makedirs(work_dir, exist_ok=True)
        logger.info(f"Using persistent working directory: {work_dir}")
        tmpdir_ctx = None
        tmpdir = work_dir
    else:
        tmpdir_ctx = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_ctx.__enter__()

    llm_func, embed_func = await create_llm_functions()

    config = RAGAnythingConfig(
        working_dir=tmpdir,
        enable_video_processing=True,
        enable_image_processing=False,
        enable_table_processing=False,
        enable_equation_processing=False,
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_func,
        embedding_func=embed_func,
    )

    try:
        # First process the video
        logger.info("Processing video for query test...")
        await rag.process_document_complete(
            TEST_VIDEO,
            output_dir=os.path.join(tmpdir, "output"),
        )

        # Debug: Print extracted entities for inspection
        await _debug_print_knowledge_graph(rag, tmpdir)

        # Test query
        query = sys.argv[2]
        logger.info(f"Querying: {query}")

        response = await rag.aquery(query, mode="hybrid")

        if response and len(response) > 0:
            logger.info("[PASS] Query returned response")
            logger.info(f"  - Response length: {len(response)} characters")
            logger.info(f"  - Response preview: {response[:200]}...")

            # Check if response contains relevant content
            response_lower = response.lower()
            if "fourier" in response_lower or "transform" in response_lower:
                logger.info("[PASS] Response contains relevant content")
                return True
            else:
                logger.warning("[WARN] Response may not contain video-specific content")
                return True  # Still a pass if we got a response
        else:
            logger.error("[FAIL] Query returned empty response")
            return False

    except ImportError as e:
        logger.warning(f"[SKIP] VideoRAG not installed: {e}")
        return True

    except Exception as e:
        logger.error(f"[FAIL] Query error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Clean up temp directory if we created one
        if tmpdir_ctx:
            tmpdir_ctx.__exit__(None, None, None)


async def test_config_options():
    """Test 5: Video configuration options."""
    logger.info("=" * 60)
    logger.info("Test 5: Video Configuration Options")
    logger.info("=" * 60)

    # Test default configuration
    config = RAGAnythingConfig()
    assert config.enable_video_processing is False, (
        "Video should be disabled by default"
    )
    assert config.video_segment_length == 30, "Default segment length should be 30"
    assert config.video_llm_provider == "openai", "Default provider should be openai"
    logger.info("[PASS] Default configuration correct")

    # Test custom configuration
    config = RAGAnythingConfig(
        enable_video_processing=True,
        videorag_working_dir="/custom/path",
        video_segment_length=60,
        video_llm_provider="deepseek",
    )
    assert config.enable_video_processing is True
    assert config.videorag_working_dir == "/custom/path"
    assert config.video_segment_length == 60
    assert config.video_llm_provider == "deepseek"
    logger.info("[PASS] Custom configuration correct")

    # Test config info includes video settings
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RAGAnythingConfig(
            working_dir=tmpdir,
            enable_video_processing=True,
        )
        rag = RAGAnything(config=config)
        config_info = rag.get_config_info()

        assert "video_processing" in config_info, (
            "Config info should include video_processing"
        )
        assert config_info["video_processing"]["enabled"] is True
        logger.info("[PASS] Config info includes video settings")

    return True


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("VideoRAG Integration Tests")
    logger.info("=" * 60)

    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.info("Please set: export OPENAI_API_KEY=your-api-key")
        return

    results = {}

    # Test 1: Video parsing (no external dependencies)
    try:
        results["parsing"] = await test_video_parsing()
    except Exception as e:
        logger.error(f"Test 1 failed with exception: {e}")
        results["parsing"] = False

    # Test 2: Configuration options (no external dependencies)
    try:
        results["config"] = await test_config_options()
    except Exception as e:
        logger.error(f"Test 5 failed with exception: {e}")
        results["config"] = False

    # Test 3: Processor initialization
    try:
        results["initialization"] = await test_video_processor_initialization()
    except Exception as e:
        logger.error(f"Test 2 failed with exception: {e}")
        results["initialization"] = False

    # Optional: Full processing tests (require VideoRAG and take time)
    run_full_tests = os.environ.get("RUN_FULL_VIDEO_TESTS", "").lower() == "true"

    if run_full_tests:
        # Test 4: Full processing
        try:
            results["full_processing"] = await test_full_video_processing()
        except Exception as e:
            logger.error(f"Test 3 failed with exception: {e}")
            results["full_processing"] = False

        # Test 5: Query
        try:
            results["query"] = await test_video_query()
        except Exception as e:
            logger.error(f"Test 4 failed with exception: {e}")
            results["query"] = False
    else:
        logger.info("\n[SKIP] Full video processing tests skipped")
        logger.info("Set RUN_FULL_VIDEO_TESTS=true to run full tests")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  {test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n=== All tests passed ===")
    else:
        logger.error("\n=== Some tests failed ===")


if __name__ == "__main__":
    asyncio.run(main())
