class FinancialParser:
    """
    Parses raw Markdown content into structured financial items.
    """
    def parse(self, markdown_content: str) -> dict:
        """
        Returns a dictionary with keys 'BS' (Balance Sheet), 'PL' (Profit Loss), 'CF' (Cash Flow).
        Each value is a list of items.
        """
        # TODO: Implement LLM-based parsing or Regex-based parsing here.
        # For now, return empty structure.
        return {
            "BS": [],
            "PL": [],
            "CF": []
        }
