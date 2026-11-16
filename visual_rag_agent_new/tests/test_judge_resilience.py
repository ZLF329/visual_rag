import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from rl.reward_judge import compute_score
from src import judge


class JudgeResilienceTests(unittest.TestCase):
    def run_batch(self, replies, answers):
        log = io.StringIO()
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only', 'DEEPSEEK_REWARD_ON_ERROR': 'raise', 'DEEPSEEK_REWARD_MAX_RETRIES': '2'}), \
             patch.object(judge, '_chat_completion', side_effect=replies) as request, \
             patch.object(judge.time, 'sleep'), redirect_stdout(log):
            scores = [compute_score('slidevqa', answer, '42', {'sample_id': 'test'}) for answer in answers]
        return scores, request.call_count, log.getvalue()

    def test_bad_json_does_not_abort_next_trajectory(self):
        scores, calls, log = self.run_batch(['', 'broken', '{}', '{"correct":true,"score":1}'], ['41', '42'])
        self.assertEqual((scores, calls), ([0.0, 1.0], 4))
        self.assertEqual(log.count('[reward_judge_error]'), 1)

    def test_transport_outage_does_not_abort_next_trajectory(self):
        scores, calls, log = self.run_batch([TimeoutError('private-details')] * 3 + ['{"correct":false,"score":0}'], ['42', '41'])
        self.assertEqual((scores, calls), ([0.0, 0.0], 4))
        self.assertEqual(log.count('[reward_judge_error]'), 1)
        self.assertNotIn('private-details', log)

    def test_mistyped_or_contradictory_verdict_is_retried(self):
        scores, calls, log = self.run_batch(['{"correct":"false","score":1}', '{"correct":true,"score":0}', '{"correct":false,"score":0}'], ['41'])
        self.assertEqual((scores, calls), ([0.0], 3))
        self.assertNotIn('[reward_judge_error]', log)

    def test_three_empty_api_replies_then_next_answer(self):
        def reply(content):
            return io.BytesIO(json.dumps({'choices': [{'message': {'content': content}}]}).encode())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'fallbacks.jsonl'
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY':'test-only', 'DEEPSEEK_FALLBACK_LOG':str(path), 'DEEPSEEK_REWARD_MAX_RETRIES':'2', 'DEEPSEEK_REWARD_ON_ERROR':'raise'}), \
                 patch.object(judge.urllib.request, 'urlopen', side_effect=[reply(''),reply(''),reply(''),reply('{"correct":true,"score":1}')]) as http, \
                 patch.object(judge.time, 'sleep'), redirect_stdout(io.StringIO()):
                self.assertEqual(compute_score('slidevqa','42','42',{'sample_id':'broken'}), 0.0)
                self.assertEqual(compute_score('slidevqa','42','42',{'sample_id':'next'}), 1.0)
                self.assertEqual(http.call_count, 4)
            event = json.loads(path.read_text())
            self.assertEqual((event['sample_id'], event['answer_reward']), ('broken', 0))
