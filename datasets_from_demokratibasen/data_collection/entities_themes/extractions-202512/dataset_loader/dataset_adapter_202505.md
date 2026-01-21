Test script results summary
1. How many documents does the dataset contain?
Total documents in raw file: 36,813
Total documents after filtering (used in processing): 34,999
Documents filtered out: 1,814 (empty text, too short, or no alphanumeric content)
2. Why are doc_types represented?
Document types categorize documents by purpose or content type. This enables:
Filtering documents by type during processing
Type-specific analysis and statistics
Different processing strategies for different document types
Quality control and validation per document type
3. How many documents of each type?
For the filtered documents (34,999 used in processing):
case_attachment: 22,761 documents (61.8%)
case_presentation: 5,844 documents (15.9%)
case_minutes: 5,844 documents (15.9%)
case_history: 975 documents (2.6%)
meeting_agenda: 750 documents (2.0%)
meeting_minutes: 639 documents (1.7%)
All documents have a dok_type field — none are missing this information.