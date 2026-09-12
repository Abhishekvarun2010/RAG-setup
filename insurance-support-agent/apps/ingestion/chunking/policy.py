"""
Structure-aware chunking strategy for policy contracts, declarations, endorsements, and schedules.

Keeps section headings, explanatory clauses, tables, and lists together as unified,
semantically coherent chunks subject to a token budget.
"""
from __future__ import annotations

from typing import Any, List, Optional

from apps.ingestion.chunking.base import BaseChunker
from apps.ingestion.models import Chunk
from apps.ingestion.structure.models import BlockType, StructuredBlock, StructuredDocument


class PolicyChunker(BaseChunker):
    """
    Chunker for insurance policy documents (contracts, declarations, endorsements, schedules).

    Rules:
    - Headings define logical section boundaries.
    - Related clauses (paragraphs), coverage tables, and lists remain intact under their heading.
    - If a section exceeds `max_tokens`, it is cleanly divided without severing tables where possible.
    - Captures `section` and `page_number` in Chunk metadata.
    """

    def chunk(self, document: StructuredDocument, **kwargs: Any) -> List[Chunk]:
        """
        Chunk a policy StructuredDocument by heading sections.

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

        def flush_current_section():
            nonlocal chunk_idx, current_heading, current_blocks
            if not current_blocks:
                return

            section_chunks = self._emit_section_chunks(
                document=document,
                section_name=current_heading or "General Details",
                blocks=current_blocks,
                start_index=chunk_idx,
                **kwargs,
            )
            chunks.extend(section_chunks)
            chunk_idx += len(section_chunks)
            current_blocks = []

        for block in document.blocks:
            if block.block_type == BlockType.HEADING:
                # 1. Prelude text before ANY heading -> merge into first heading
                if current_heading is None and current_blocks:
                    current_heading = block.content.strip()
                    current_blocks.append(block)
                    continue

                # 2. Company banner title (e.g. "Meridian Mutual Insurance SE") -> merge into "Policy Declarations"
                if current_heading and "insurance se" in current_heading.lower() and len(current_blocks) <= 2:
                    current_heading = block.content.strip()
                    current_blocks.append(block)
                    continue

                # 3. Regular heading boundary: flush previous section
                flush_current_section()
                current_heading = block.content.strip()
                current_blocks.append(block)
            else:
                current_blocks.append(block)

        # Flush final section
        flush_current_section()

        return chunks

    def _emit_section_chunks(
        self,
        document: StructuredDocument,
        section_name: str,
        blocks: List[StructuredBlock],
        start_index: int,
        **kwargs: Any,
    ) -> List[Chunk]:
        """
        Produce one or more Chunks from a section's blocks, splitting if total
        tokens exceed `self.max_tokens`.
        """
        page_num = blocks[0].page_number if blocks else 1

        # Format full section content
        content_parts: List[str] = []
        for b in blocks:
            if b.block_type == BlockType.HEADING:
                content_parts.append(f"### {b.content.strip()}")
            else:
                content_parts.append(b.content.strip())

        full_content = "\n\n".join(content_parts).strip()
        estimated = self.estimate_tokens(full_content)

        # If fits comfortably within token limit, emit as single cohesive chunk
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

        # Otherwise, split across blocks while preserving the section header
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
                # Flush current batch
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
