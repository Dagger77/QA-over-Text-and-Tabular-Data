import unittest
import asyncio
from agents.sql_agent import run_sql_agent
from tests.utils import missing_required_tables


class TestSQLAgent(unittest.TestCase):

    def test_valid_query(self):
        if missing_required_tables():
            self.skipTest("Required tables are missing in the database.")
        result = asyncio.run(run_sql_agent("How many female students scored above 90 in math?"))
        self.assertIn("rows", result)
        self.assertIsInstance(result["rows"], list)

    def test_invalid_column(self):
        if missing_required_tables():
            self.skipTest("Required tables are missing in the database.")
        result = asyncio.run(run_sql_agent("How many students have a pet?"))
        self.assertTrue("error" in result or "explanation" in result)


if __name__ == '__main__':
    unittest.main()
