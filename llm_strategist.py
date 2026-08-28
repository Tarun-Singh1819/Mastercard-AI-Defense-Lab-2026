"""
MASTERCARD AI DEFENSE LAB — LLM Red Team Controller
Acts as the Strategy Advisor for the Red Team.
"""

import random

class LLMRedTeamController:
    """
    Takes mathematical blind spots from SmartEvolution and outputs 
    a strategic narrative + parameter overrides to evade the Blue Team.
    """
    
    def __init__(self, use_mock=True):
        self.use_mock = use_mock
        
    def generate_strategy(self, epoch, blind_spots, current_params):
        """
        Outputs a strategic narrative and parameter overrides.
        """
        if not blind_spots:
            return {
                "narrative": "🤖 AI Strategist: No blind spots detected. The Blue Team is highly effective. Shifting to random topology rotation.",
                "overrides": {}
            }
            
        if self.use_mock:
            return self._mock_llm_response(epoch, blind_spots, current_params)
        else:
            return self._real_llm_response(epoch, blind_spots, current_params)
            
    def _mock_llm_response(self, epoch, blind_spots, current_params):
        overrides = {}
        strategy_points = []
        
        if 'device_shared_count' in blind_spots or 'ip_shared_count' in blind_spots:
            strategy_points.append("Blue Team is successfully clustering mules using device/IP graphs.")
            # Evade by reducing device sharing
            overrides['device_sharing_ratio'] = random.uniform(0.05, 0.15)
            strategy_points.append("Action: Drastically reduce device sharing (down to ~10%) to blend in with legitimate users.")
            
        if 'pagerank' in blind_spots:
            strategy_points.append("Centrality (PageRank) is exposing our aggregation accounts.")
            # Evade by using dispersive topologies
            overrides['topology'] = ['star_burst', 'cross_rail']
            strategy_points.append("Action: Abandon Fan-Out-Fan-In layering. Use Star Burst topologies to disperse funds without creating highly ranked hubs.")
            
        if 'in_degree' in blind_spots or 'out_degree' in blind_spots:
            strategy_points.append("Node degree signatures are anomalous.")
            # Evade by increasing noise txns
            overrides['noise_txn_per_mule'] = random.randint(4, 8)
            strategy_points.append("Action: Inject 4-8 legitimate 'noise' transactions per mule to obscure the laundering flow and normalize node degrees.")
            
        if not overrides:
            strategy_points.append("General evasion mode activated.")
            overrides['amount_mean'] = random.uniform(15000, 30000)
            strategy_points.append("Action: Shift average laundering amounts to evade historical transaction thresholds.")
            
        narrative = f"🤖 AI STRATEGIST REPORT (Epoch {epoch}):\n  " + "\n  ".join(strategy_points)
        
        return {
            "narrative": narrative,
            "overrides": overrides
        }
        
    def _real_llm_response(self, epoch, blind_spots, current_params):
        """
        Uses Google Gemini to generate the strategy.
        Requires 'google-generativeai' package.
        """
        try:
            import google.generativeai as genai
            import os
            
            # TODO: Add your Gemini API key here or set it as an environment variable
            api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            You are the AI Strategist for a Red Team money laundering operation.
            Your goal is to evade the Blue Team's Graph Neural Network (GNN).
            
            Current Epoch: {epoch}
            Current Attack Parameters: {current_params}
            
            The Blue Team successfully detected some of our fraud. We analyzed their blind spots (where they missed our fraud).
            Blind Spots Analysis: {blind_spots}
            
            Based on these blind spots, provide a strategic narrative on how we should change our attack parameters to evade detection next time.
            Return ONLY a JSON object with two keys:
            1. "narrative": A 2-3 sentence strategic explanation.
            2. "overrides": A dictionary of parameters to change (e.g., {{"device_sharing_ratio": 0.1, "topology": ["star_burst"]}}).
            """
            
            response = model.generate_content(prompt)
            # Basic parsing (assumes the model returns clean JSON)
            import json
            import re
            
            text = response.text
            # Extract JSON block if surrounded by markdown ticks
            match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
            if match:
                text = match.group(1)
                
            result = json.loads(text)
            
            narrative = f"🤖 GEMINI STRATEGIST REPORT (Epoch {epoch}):\n  {result.get('narrative', '')}"
            return {
                "narrative": narrative,
                "overrides": result.get('overrides', {})
            }
            
        except Exception as e:
            return {
                "narrative": f"🤖 LLM API Error: {str(e)}. Falling back to mock strategy.",
                "overrides": {}
            }
