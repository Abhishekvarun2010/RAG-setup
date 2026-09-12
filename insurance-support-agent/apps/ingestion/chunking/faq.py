"""
Question-Answer chunking strategy for Customer FAQs and knowledge manuals.

Ensures questions and answers stay together in every chunk, repeating the question
in split chunks so that each chunk is 100% understandable independently.
"""
from __future__ import annotations

from typing import Any, List, Optional

from apps.ingestion.chunking.base import BaseChunker
from apps.ingestion.models import Chunk
from apps.ingestion.structure.models import BlockType, StructuredBlock, StructuredDocument


class FAQChunker(BaseChunker):
    """
    Chunker for FAQs and Q&A knowledge base documents.

    Rules:
    - Questions (headings or question blocks) are paired directly with their answers.
    - Output format:
        Question: {Question}

        Answer:
        {Answer content}
    - If an answer exceeds `max_tokens`, it is split and the Question is repeated
      in each subsequent chunk (`Answer (Part 2)`).
    """

    def chunk(self, document: StructuredDocument, **kwargs: Any) -> List[Chunk]:
        """
        Chunk an FAQ StructuredDocument into Question-Answer pairs.

        Args:
            document: StructuredDocument with classified blocks.
            **kwargs: Extra attributes or overrides.

        Returns:
            List of self-contained Q&A Chunk objects.
        """
        if not document.blocks:
            return []

        chunks: List[Chunk] = []
        chunk_idx = 1

        current_question: Optional[str] = None
        current_answer_blocks: List[StructuredBlock] = []
        question_page: int = 1

        def flush_qa():
            nonlocal chunk_idx, current_question, current_answer_blocks, question_page
            if not current_question and not current_answer_blocks:
                return

            if current_question:
                qa_chunks = self._emit_qa_chunks(
                    document=document,
                    question=current_question,
                    answer_blocks=current_answer_blocks,
                    page_number=question_page,
                    start_index=chunk_idx,
                    **kwargs,
                )
                chunks.extend(qa_chunks)
                chunk_idx += len(qa_chunks)
            elif current_answer_blocks:
                # Prelude / intro text before the first question
                prelude_content = "\n\n".join(b.content.strip() for b in current_answer_blocks).strip()
                if len(prelude_content) > 30 and not prelude_content.startswith("<!--"):
                    chunk = self._create_chunk(
                        document=document,
                        content=prelude_content,
                        section="FAQ Overview",
                        page_number=current_answer_blocks[0].page_number,
                        chunk_index=chunk_idx,
                        **kwargs,
                    )
                    chunks.append(chunk)
                    chunk_idx += 1

            current_question = None
            current_answer_blocks = []

        for block in document.blocks:
            is_question = self._is_question_block(block)

            if is_question:
                flush_qa()
                current_question = self._clean_question_text(block.content)
                question_page = block.page_number
            else:
                current_answer_blocks.append(block)

        flush_qa()
        return chunks

    def _is_question_block(self, block: StructuredBlock) -> bool:
        """Identify whether a block represents a question."""
        text = block.content.strip()
        if block.block_type == BlockType.HEADING:
            # Skip document title if it is just "Customer FAQ"
            if text.lower() in {"customer faq", "faq", "frequently asked questions"}:
                return False
            return True
        # Paragraph ending in '?' or starting with 'Q:'
        if text.endswith("?") or text.lower().startswith(("q:", "question:")):
            return True
        return False

    @staticmethod
    def _clean_question_text(raw_text: str) -> str:
        """Normalize question text."""
        cleaned = raw_text.strip().lstrip("#").strip()
        if cleaned.lower().startswith("question:"):
            cleaned = cleaned[9:].strip()
        elif cleaned.lower().startswith("q:"):
            cleaned = cleaned[2:].strip()
        return cleaned

    def _emit_qa_chunks(
        self,
        document: StructuredDocument,
        question: str,
        answer_blocks: List[StructuredBlock],
        page_number: int,
        start_index: int,
        **kwargs: Any,
    ) -> List[Chunk]:
        """Format Question-Answer pair and split across chunks if answer exceeds max_tokens."""
        answer_parts: List[str] = [b.content.strip() for b in answer_blocks if b.content.strip()]
        full_answer = "\n\n".join(answer_parts).strip()

        if not full_answer:
            full_answer = "[No answer provided]"

        combined_text = f"Question: {question}\n\nAnswer:\n{full_answer}"
        estimated_tokens = self.estimate_tokens(combined_text)

        # Fits within token limit
        if estimated_tokens <= self.max_tokens or len(answer_blocks) <= 1:
            return [
                self._create_chunk(
                    document=document,
                    content=combined_text,
                    section=question,
                    page_number=page_number,
                    chunk_index=start_index,
                    **kwargs,
                )
            ]

        # Multi-part answer: repeat Question in every chunk
        sub_chunks: List[Chunk] = []
        curr_answer_parts: List[str] = []
        curr_tokens = self.estimate_tokens(f"Question: {question}\n\nAnswer:\n")
        part_num = 1
        sub_idx = start_index

        for b in answer_blocks:
            part_text = b.content.strip()
            part_tokens = self.estimate_tokens(part_text)

            if curr_tokens + part_tokens > self.max_tokens and curr_answer_parts:
                answer_content = "\n\n".join(curr_answer_parts).strip()
                label = f"Answer (Part {part_num}):" if part_num > 1 else "Answer:"
                chunk_text = f"Question: {question}\n\n{label}\n{answer_content}"
                sub_chunks.append(
                    self._create_chunk(
                        document=document,
                        content=chunk_text,
                        section=question,
                        page_number=page_number,
                        chunk_index=sub_idx,
                        **kwargs,
                    )
                )
                sub_idx += 1
                part_num += 1
                curr_answer_parts = [part_text]
                curr_tokens = self.estimate_tokens(f"Question: {question}\n\nAnswer (Part {part_num}):\n") + part_tokens
            else:
                curr_answer_parts.append(part_text)
                curr_tokens += part_tokens

        if curr_answer_parts:
            answer_content = "\n\n".join(curr_answer_parts).strip()
            label = f"Answer (Part {part_num}):" if part_num > 1 else "Answer:"
            chunk_text = f"Question: {question}\n\n{label}\n{answer_content}"
            sub_chunks.append(
                self._create_chunk(
                    document=document,
                    content=chunk_text,
                    section=question,
                    page_number=page_number,
                    chunk_index=sub_idx,
                    **kwargs,
                )
            )

        return sub_chunks
