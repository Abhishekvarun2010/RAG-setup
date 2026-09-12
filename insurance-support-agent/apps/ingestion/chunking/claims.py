"""
Structure-aware chunking strategy for claims documents.

Preserves logical sections in FNOLs, adjuster reports, estimates, and settlement/denial
letters without arbitrarily fragmenting loss descriptions or determination findings.
"""
from __future__ import annotations

from typing import Any, List, Optional

from apps.ingestion.chunking.base import BaseChunker
from apps.ingestion.models import Chunk
from apps.ingestion.structure.models import BlockType, StructuredBlock, StructuredDocument


class ClaimsChunker(BaseChunker):
    """
    Chunker for insurance claim reports (FNOL, adjuster reports, estimates, settlement letters).

    Rules:
    - Logical claim sections (Claim Details, Loss Description, Findings, Determination)
      are preserved together as cohesive chunks.
    - Captures `claim_id`, `policy_id`, `section`, and `page_number`.
    - Merges small adjacent key-value fields so they do not produce single-line fragmented chunks.
    """

    def chunk(self, document: StructuredDocument, **kwargs: Any) -> List[Chunk]:
        """
        Chunk a claims StructuredDocument by logical report sections.

        Args:
            document: StructuredDocument with classified blocks.
            **kwargs: Extra attributes or overrides.

        Returns:
            List of structured Chunk objects.
        """
        if not document.blocks:
            return []

        chunks: List[Chunk] = []
        chunk_idx = 1

        current_heading: Optional[str] = None
        current_blocks: List[StructuredBlock] = []

        def flush_section():
            nonlocal chunk_idx, current_heading, current_blocks
            if not current_blocks:
                return

            section_chunks = self._emit_claims_chunks(
                document=document,
                section_name=current_heading or "Claim Details",
                blocks=current_blocks,
                start_index=chunk_idx,
                **kwargs,
            )
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)
            current_blocks = []

        for block in document.blocks:
            if block.block_type == BlockType.HEADING:
                text = block.content.strip()
                # Skip company banner header (e.g. Meridian Mutual Insurance SE)
                if "insurance se" in text.lower() or "synthetic" in text.lower():
                    current_blocks.append(block)
                    continue

                flush_section()
                current_heading = text
                current_blocks.append(block)
            else:
                current_blocks.append(block)

        flush_section()
        return chunks

    def _emit_claims_chunks(
        self,
        document: StructuredDocument,
        section_name: str,
        blocks: List[StructuredBlock],
        start_index: int,
        **kwargs: Any,
    ) -> List[Chunk]:
        """Emit one or more chunks for a claims section, adhering to token budget."""
        page_num = blocks[0].page_number if blocks else 1

        content_parts: List[str] = []
        for b in blocks:
            if b.block_type == BlockType.HEADING:
                content_parts.append(f"### {b.content.strip()}")
            else:
                content_parts.append(b.content.strip())

        full_content = "\n\n".join(content_parts).strip()
        estimated = self.estimate_tokens(full_content)

        if estimated <= self.max_tokens or len(blocks) <= 2:
            return [
                self._create_chunk(
                    document=document,
                    content=full_content,
                    section=section_name,
                    page_number=page_num,
                    chunk_index=start_index,
                    **kwargs,
                )
            ]

        # Split across blocks if overly long
        sub_chunks: List[Chunk] = []
        curr_parts: List[str] = [f"### {section_name}"]
        curr_tokens = self.estimate_tokens(curr_parts[0])
        sub_idx = start_index

        for b in blocks:
            if b.block_type == BlockType.HEADING and b.content.strip() == section_name:
                continue

            part_text = b.content.strip()
            part_tokens = self.estimate_tokens(part_text)

            if curr_tokens + part_tokens > self.max_tokens and len(curr_parts) > 1:
                sub_chunks.append(
                    self._create_chunk(
                        document=document,
                        content="\n\n".join(curr_parts).strip(),
                        section=section_name,
                        page_number=page_num,
                        chunk_index=sub_idx,
                        **kwargs,
                    )
                )
                sub_idx += 1
                curr_parts = [f"### {section_name} (Continued)", part_text]
                curr_tokens = self.estimate_tokens(curr_parts[0]) + part_tokens
            else:
                curr_parts.append(part_text)
                curr_tokens += part_tokens

        if curr_parts and (len(curr_parts) > 1 or curr_parts[0] != f"### {section_name}"):
            sub_chunks.append(
                self._create_chunk(
                    document=document,
                    content="\n\n".join(curr_parts).strip(),
                    section=section_name,
                    page_number=page_num,
                    chunk_index=sub_idx,
                    **kwargs,
                )
            )

        return sub_chunks
