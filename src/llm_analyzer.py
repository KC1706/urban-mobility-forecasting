"""
LLM Analyzer: Use LLMs to analyze urban mobility patterns
Generates insights about congestion, routes, accessibility, and policy
Supports: OpenAI, Anthropic, HuggingFace, Local LLMs
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class MobilityLLMAnalyzer:
    """Use LLM to analyze mobility data and generate insights"""
    
    def __init__(self, llm_model="gpt-4", api_key=None, provider="openai"):
        """
        Initialize LLM analyzer
        
        Args:
            llm_model: Model name (gpt-4, meta-llama/Llama-2-7b, mistralai/Mistral-7B, etc.)
            api_key: API key for LLM service (if None, read from env)
            provider: LLM provider ('openai', 'anthropic', 'huggingface', 'local')
        """
        self.llm_model = llm_model
        self.provider = provider
        self.api_key = api_key or self._get_api_key()
        self.client = None
        self.model = None  # For local transformers models
        self.tokenizer = None  # For local models
        self._initialize_client()
    
    def _get_api_key(self):
        """Get API key based on provider"""
        if self.provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.provider == "huggingface":
            return os.getenv("HUGGINGFACE_API_KEY")
        elif self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return None
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        try:
            if self.provider == "openai":
                if not self.api_key:
                    logger.warning(f"No API key found for {self.provider}. Running in demo mode.")
                    return
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"✓ Initialized OpenAI: {self.llm_model}")
                
            elif self.provider == "huggingface":
                if not self.api_key:
                    logger.warning(f"No API key found for {self.provider}. Running in demo mode.")
                    return
                from huggingface_hub import InferenceClient
                self.client = InferenceClient(api_key=self.api_key)
                logger.info(f"✓ Initialized HuggingFace: {self.llm_model}")
                
            elif self.provider == "anthropic":
                if not self.api_key:
                    logger.warning(f"No API key found for {self.provider}. Running in demo mode.")
                    return
                from anthropic import Anthropic
                self.client = Anthropic(api_key=self.api_key)
                logger.info(f"✓ Initialized Anthropic: {self.llm_model}")
                
            elif self.provider == "local":
                # Load local HuggingFace model using transformers
                logger.info(f"Loading local HuggingFace model: {self.llm_model}")
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch
                    
                    # Detect device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    # Load tokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
                    
                    # Load model with optimizations
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.llm_model,
                        device_map="auto",
                        dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.model.eval()
                    logger.info(f"✓ Loaded local model: {self.llm_model} on {device}")
                except ImportError:
                    logger.error("transformers, torch required for local models")
                    logger.error("Install with: pip install transformers torch")
                    
        except ImportError as e:
            logger.error(f"Package not installed: {e}")
            logger.error(f"Install with: pip install {self.provider}")
    
    def load_analysis_dataset(self) -> Dict[str, Any]:
        """Load processed mobility dataset"""
        dataset_file = PROCESSED_DATA_DIR / "analysis_dataset.json"
        
        if not dataset_file.exists():
            logger.error(f"Dataset not found: {dataset_file}")
            return {}
        
        with open(dataset_file) as f:
            return json.load(f)
    
    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> str:
        """
        Call LLM with provider-specific formatting
        
        Supports: OpenAI, HuggingFace, Anthropic, Local
        """
        if not self.client and not self.model:
            return None
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif self.provider == "huggingface":
                # HuggingFace Inference API
                response = self.client.text_generation(
                    prompt=f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
                    model=self.llm_model,
                    max_new_tokens=max_tokens,
                    temperature=0.7
                )
                return response
            
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.llm_model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return response.content[0].text
            
            elif self.provider == "local":
                # Use local transformers model
                import torch
                
                prompt = f"{system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"
                
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
                
                # Generate with greedy decoding (more stable than sampling)
                with torch.no_grad():
                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=min(max_tokens, 256),
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                # Decode
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                # Remove the prompt from the response
                response = response[len(prompt):].strip()
                return response
        
        except Exception as e:
            logger.error(f"LLM API error ({self.provider}): {e}")
            return None
    
    def generate_congestion_summary(self, taxi_data: Dict) -> str:
        """Generate LLM summary of congestion patterns from taxi data"""
        
        avg_trip_dur = taxi_data.get('avg_trip_duration_min', 'N/A')
        if not isinstance(avg_trip_dur, str):
            avg_trip_dur = f"{avg_trip_dur:.1f}"
        
        summary_text = f"""NYC TAXI CONGESTION DATA ANALYSIS - STRUCTURED FORMAT
=====================================================

CONTEXT:
You are analyzing NYC taxi congestion data to generate actionable insights for urban planners.
Focus on: (1) Specific, measurable findings (2) Equity impact (3) Implementable solutions

DATA INPUTS:
- Total trips analyzed: {taxi_data.get('total_trips', 'N/A')}
- Average trip duration: {avg_trip_dur} minutes
- Peak traffic hours: {taxi_data.get('peak_hour', 'N/A')}:00 (rush hours)
- Dataset: January 2024 NYC taxi patterns

STRUCTURED ANALYSIS (Follow this format exactly):

1. CONGESTION SEVERITY & HOTSPOTS
   Format: [Neighborhood]: [Hours lost/day] hours, [Root cause]
   - Most congested area and severity
   - Secondary hotspots (rank by impact)
   - Geographic patterns (cross-town, bridge access, etc)

2. EQUITY IMPACT ASSESSMENT  
   Format: [Demographic group]: [Impact description] → [% affected]
   - Which communities face longest delays
   - Link to job/healthcare/school access gaps
   - Quantify if possible

3. TEMPORAL BREAKDOWN
   - Morning peak (hours + % increase)
   - Evening peak (hours + % increase)
   - Off-peak patterns

4. ROOT CAUSE ANALYSIS
   For each hotspot: Identify 2-3 primary causes (infrastructure/demand/policy)

5. SOLUTION RECOMMENDATIONS (HIGHEST PRIORITY FIRST)
   Format: 
   [Action] → Expected improvement [%] → Cost [Low/Medium/High] → Timeline [weeks]
   • Include equity considerations
   • Address implementation barriers
   • Include enforcement/monitoring needs

6. EXPECTED OUTCOMES
   - Total commute hours saved daily
   - Economic value (estimated $)
   - Equity improvement (% gaining access)"""
        
        if not self.client and not self.model:
            logger.warning("LLM client not initialized. Using demo output.")
            return self._generate_demo_congestion_analysis(taxi_data)
        
        system_prompt = """You are Dr. Marcus Chen, Senior Urban Mobility Analyst at George Mason University.
Expertise: NYC transit equity, congestion dynamics, data-driven policy analysis.

Your approach:
• Combine quantitative rigor with actionable insights
• Always lead with equity concerns and vulnerable populations
• Prioritize feasible, high-impact recommendations
• Acknowledge trade-offs and political realities
• Use specific NYC location names (neighborhoods, major streets, routes)
• Include numbers: times, percentages, costs where possible
• Focus on findings that matter for decision-makers"""
        try:
            response = self._call_llm(system_prompt, summary_text, max_tokens=1200)
            
            if response:
                return response
            return self._generate_demo_congestion_analysis(taxi_data)

        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_demo_congestion_analysis(taxi_data)
    
    def generate_transit_accessibility_report(self, gtfs_data: Dict) -> str:
        """Analyze transit accessibility from GTFS data"""
        
        avg_stops = gtfs_data.get('subway', {}).get('avg_stops_per_route', 'N/A')
        if not isinstance(avg_stops, str):
            avg_stops = f"{avg_stops:.1f}"
        
        analysis_prompt = f"""NYC MULTI-MODAL TRANSIT NETWORK EQUITY ANALYSIS
==============================================

NETWORK INVENTORY:

Subway System:
- Total stops: {gtfs_data.get('subway', {}).get('num_stops', 'N/A')}
- Routes: {gtfs_data.get('subway', {}).get('num_routes', 'N/A')}
- Avg stations per route: {avg_stops}
- Coverage: Manhattan-centric with outer borough gaps

Bus System:
- Total stops: {gtfs_data.get('bus', {}).get('num_stops', 'N/A')}
- Routes: {gtfs_data.get('bus', {}).get('num_routes', 'N/A')}
- Primary role: First/last-mile, cross-town, outer borough access

Railroad (LIRR, Metro-North):
- Stops: {gtfs_data.get('railroad', {}).get('num_stops', 'N/A')}
- Routes: {gtfs_data.get('railroad', {}).get('num_routes', 'N/A')}
- Focus: Commuter rail, outer neighborhoods

EQUITY ANALYSIS FRAMEWORK:

1. GEOGRAPHIC COVERAGE GAPS
   • Which neighborhoods lack subway access? (prioritize low-income areas)
   • Distance to nearest subway station by neighborhood
   • Outer borough vs. Manhattan service quality comparison
   • Accessibility barriers: physical (stairs), information (language), cost

2. SERVICE EQUITY & FREQUENCY
   • Route frequency comparison (peak vs. off-peak)
   • Late-night service gaps (affects shift workers)
   • Weekend service quality (affects low-income leisure activities)
   • ADA accessibility: wheelchair access, elevator availability

3. MULTI-MODAL CONNECTIVITY
   • Subway-to-bus transfer hubs and wait times
   • First/last-mile solutions (bike, scooter, walking distance)
   • Cross-borough mobility barriers
   • Transit deserts: underserved neighborhoods and why

4. UNDERSERVED POPULATIONS ANALYSIS
   • Low-income communities with limited transit options
   • Areas with high car-free households (constrained by service)
   • Senior/disability accessibility gaps
   • Communities of color with historically fewer transit investments

5. RECOMMENDATIONS BY PRIORITY & EQUITY IMPACT
   • Bus service increases to underserved areas (quick wins)
   • Subway accessibility improvements (physical access)
   • Late-night service for essential workers
   • Improved transfer experience (time, safety, clarity)
   • Long-term expansion to transit deserts
   
   Include: Cost estimates, expected ridership impact, equity metrics

6. SPECIFIC CHALLENGES IN NYC CONTEXT
   • Outer boroughs: Queens, Bronx, Staten Island coverage gaps
   • Manhattan: over-reliance on crowded core lines
   • Aging infrastructure requiring capital investment
   • Funding constraints and prioritization framework"""
        
        if not self.client and not self.model:
            logger.warning("LLM client not initialized. Using demo output.")
            return self._generate_demo_transit_analysis(gtfs_data)
        
        system_prompt = """You are Dr. Sarah Johnson, Transportation Equity Researcher at George Mason University.
Expertise: Transit access, environmental justice, urban equity, NYC transit planning.

Your approach:
• Equity-first analysis: Who benefits? Who is left behind?
• Specific neighborhood analysis (use real NYC names)
• Quantified accessibility metrics (% coverage, distance, service frequency)
• Intersectional perspective: race, income, age, disability status
• Identify "transportation disadvantaged" populations
• Propose equity-centered solutions first
• Acknowledge systemic barriers and budget constraints"""
        try:
            response = self._call_llm(system_prompt, analysis_prompt, max_tokens=1200)
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_demo_transit_analysis(gtfs_data)
    
    def scenario_analysis(self, scenario: str) -> str:
        """
        Perform what-if scenario analysis using LLM
        
        Scenarios:
        - "close_line_L": Impact of closing L train
        - "reduce_bus_service": 20% bus service reduction
        - "add_bike_lanes": Expand bike infrastructure
        """
        
        scenario_details = {
            "close_line_L": {
                "title": "L TRAIN SERVICE CLOSURE SCENARIO",
                "context": """CONTEXT: The L subway line (14th Street Crosstown) serves 20 stations, 
connects Brooklyn (Canarsie) to Manhattan with 200,000+ daily riders.
It is a critical first/last-mile and cross-borough connector.
Closure could last 2-5 years (infrastructure maintenance).""",
                "impact_areas": [
                    "Williamsburg, Brooklyn → Manhattan commuters (high-income but transit-dependent)",
                    "East Village, LES → Flatbush avenue corridor disruption",
                    "1st Ave bus routes (severe overcrowding expected)",
                    "Bike commuting surge (environmental opportunity)",
                    "Real estate impacts: Williamsburg rents likely to decrease"
                ]
            },
            "reduce_bus_service": {
                "title": "20% BUS SERVICE REDUCTION SCENARIO",
                "context": """CONTEXT: NYC buses serve 5.5+ million daily riders, 70% low-income.
Bus is primary transit for outer boroughs and essential workers.
20% reduction = ~700,000 riders losing access or facing 50%+ longer waits.""",
                "impact_areas": [
                    "Essential workers: healthcare, sanitation, retail (shift times)",
                    "Low-income families without car access (disproportionate impact)",
                    "Outer boroughs: Bronx, Eastern Queens, southern Brooklyn",
                    "Cross-town routes (least profitable, highest need)",
                    "Late-night service elimination (affects night shift workers)"
                ]
            },
            "add_bike_lanes": {
                "title": "BIKE INFRASTRUCTURE EXPANSION SCENARIO",
                "context": """CONTEXT: NYC has 600+ miles of bike lanes. Protected lanes increase usage 5-10x.
Opportunity: First/last-mile solution, reduce car congestion, climate benefits.""",
                "impact_areas": [
                    "East Side: FDR Drive lane reallocation (reduced car throughput)",
                    "Brooklyn: Prospect Park connections (recreation + commuting)",
                    "Cross-town gaps: 14th, 23rd, 34th streets (commuter corridors)",
                    "Safety benefits: 20-30% reduction in bike injuries (protected lanes)",
                    "Equity: Gentrification risk in emerging neighborhoods"
                ]
            },
            "fare_increase": {
                "title": "15% TRANSIT FARE INCREASE SCENARIO",
                "context": """CONTEXT: Current MTA fare is $2.90 base. 15% increase → $3.34.
Affects 5.7M daily riders. Regressive tax (% income impact highest for poor).""",
                "impact_areas": [
                    "Low-income riders: Choice between transit, food, healthcare",
                    "Ridership loss: Expect 3-8% reduction, peak off-peak shift",
                    "Revenue: +$100M+ annually, but long-term ridership erosion",
                    "Equity impact: Regressive (poor pay higher % of income)",
                    "Alternative modes: More car use, gig bikes, walking avoidance"
                ]
            }
        }
        
        scenario_info = scenario_details.get(scenario, {
            "title": scenario.upper(),
            "context": f"Analyzing: {scenario}",
            "impact_areas": []
        })
        
        analysis_prompt = f"""SCENARIO PLANNING: {scenario_info['title']}
===========================================================

{scenario_info['context']}

AFFECTED POPULATIONS & IMPACT AREAS:
{chr(10).join(f"• {area}" for area in scenario_info['impact_areas'])}

COMPREHENSIVE IMPACT ANALYSIS REQUIRED:

1. DIRECT IMPACTS (Immediate, quantifiable)
   • Ridership changes (% loss/gain by mode)
   • Travel time increases (minutes added per trip)
   • Cost burden (especially for low-income riders)
   • Specific neighborhoods most affected
   • Essential worker impacts (healthcare, retail, sanitation)

2. EQUITY CONSEQUENCES
   • Regressive vs progressive impact (who loses most?)
   • Communities of color disproportionately impacted? (Yes/No + why)
   • Elderly and disabled rider challenges
   • Geographic equity: Outer boroughs vs Manhattan
   • Income distribution of affected riders

3. SYSTEM CASCADES
   • Mode shifts: What happens to displaced riders?
   • Network effects: Do alternative routes work?
   • Congestion ripple effects (if car mode increase)
   • Air quality and public health implications
   • Real estate market shifts

4. POLICY MITIGATION OPTIONS
   • Equity-centered compensation (who gets what?)
   • Alternative route investments
   • Pricing mechanisms (congestion pricing, subsidies)
   • Timeline and transition support
   • Cost-benefit analysis

5. QUANTIFIED FORECASTS (Use reasonable estimates)
   • Total cost to riders (annual $ loss)
   • Revenue impact (positive/negative)
   • CO2 impact (if mode shift to cars)
   • Public health (physical activity, air quality)
   • Economic productivity (congestion hours lost)

6. POLITICAL REALITIES & CONSTRAINTS
   • Which interest groups oppose/support?
   • Feasibility given NYC political environment
   • Equity trade-offs: Who bears the cost?
   • Realistic implementation timeline"""
        
        if not self.client and not self.model:
            logger.warning("LLM client not initialized. Using demo output.")
            return self._generate_demo_scenario(scenario)
        
        system_prompt = """You are Dr. Michael Torres, Urban Policy Analyst & Transportation Economist.
Expertise: Scenario planning, equity impact analysis, NYC transportation policy, political feasibility.

Your approach:
• Balance technical analysis with political reality
• Lead with equity: Who bears the costs vs who benefits?
• Quantify impacts: dollars, minutes, percentages, CO2, lives
• Acknowledge trade-offs: No perfect solutions
• Frame recommendations around: Feasibility, timeline, cost, equity
• Use specific NYC neighborhoods and demographics
• Distinguish likely vs speculative impacts
• Consider unintended consequences"""
        
        try:
            response = self._call_llm(system_prompt, analysis_prompt, max_tokens=1400)
            
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._generate_demo_scenario(scenario)
    
    def _generate_demo_congestion_analysis(self, taxi_data):
        """Demo output when LLM not available"""
        avg_dur = taxi_data.get('avg_trip_duration_min', 'N/A')
        avg_dur_str = avg_dur if isinstance(avg_dur, str) else f"{avg_dur:.1f}"
        return f"""
        CONGESTION ANALYSIS - NYC TAXI DATA
        
        Key Findings:
        - Estimated {taxi_data.get('total_trips', 'N/A')} taxi trips analyzed
        - Average trip duration: {avg_dur_str} minutes
        - Peak hour: {taxi_data.get('peak_hour', 'N/A')}:00 (lunch/evening commute)
        
        Congestion Patterns:
        1. Midtown Manhattan (5th-8th Ave): 15-20 min delays 09:00-10:00, 17:00-19:00
        2. Downtown: Financial District gridlock 08:00-09:30
        3. Bridge crossings: Williamsburg, Manhattan bridges peak 08:00-09:00
        
        Route Efficiency:
        - Cross-town routes 15% slower than uptown/downtown
        - East Side Highway preferred for speed
        
        Recommendations:
        1. Implement congestion pricing in Midtown (15% reduction projected)
        2. Expand bus lanes on cross-town streets
        3. Optimize traffic signal timing (rush hours)
        4. Promote off-peak travel incentives
        """
    
    def _generate_demo_transit_analysis(self, gtfs_data):
        """Demo transit analysis"""
        return f"""
        TRANSIT ACCESSIBILITY ANALYSIS - NYC MTA
        
        Network Coverage:
        - Subway: {gtfs_data.get('subway', {}).get('num_stops', 'N/A')} stations across {gtfs_data.get('subway', {}).get('num_routes', 'N/A')} routes
        - Bus: {gtfs_data.get('bus', {}).get('num_stops', 'N/A')} stops on {gtfs_data.get('bus', {}).get('num_routes', 'N/A')} routes
        - LIRR: {gtfs_data.get('railroad', {}).get('num_stops', 'N/A')} stations
        
        Accessibility Gaps:
        1. Outer boroughs (Queens, Bronx): Lower subway coverage, bus-dependent
        2. Underserved areas: Eastern Queens, South Brooklyn
        3. Late-night service: Limited after midnight
        
        Multi-Modal Connectivity:
        - Good: Jamaica, Atlantic Terminal hubs
        - Weak: Outer neighborhood connections
        
        Recommendations:
        1. Increase bus frequency in underserved areas (15% funding increase)
        2. Extend subway to outer neighborhoods (long-term capital plan)
        3. Improve bus-subway transfer experience
        4. Expand night service for essential workers
        """
    
    def _generate_demo_scenario(self, scenario):
        """Demo scenario analysis"""
        scenarios_output = {
            "close_line_L": """
            SCENARIO: NYC MTA L Train Closure
            
            Impacts:
            - 225,000 daily riders affected
            - Commute times: +15-25 minutes for displaced riders
            - Alternative routes: M14 bus +40%, 1/2/3 lines +20%
            - Congestion increase: 12% in East Village/Williamsburg
            
            Equity Impact:
            - Low-income riders in outer Williamsburg hit hardest
            - Mitigation: Free transfers, express bus service
            
            Long-term: Consider L-train modernization vs. permanent closure
            """,
            "reduce_bus_service": """
            SCENARIO: 20% Bus Service Reduction
            
            Effects:
            - 300,000 daily trips eliminated
            - Wait times: +8-12 minutes
            - Ridership loss: 18-22%
            - Job accessibility: Disproportionate impact on outer boroughs
            
            Revenue/Cost:
            - Savings: $200M annually
            - Lost fares: $80M (net savings $120M)
            - Social cost: $300M+ (job loss, healthcare delays)
            
            Recommendation: Targeted reductions in low-demand times, not peak service
            """,
            "add_bike_lanes": """
            SCENARIO: Expansion of Protected Bike Infrastructure
            
            Benefits:
            - 50,000 new daily bike trips (50% of barrier reduction)
            - Car congestion: 3-5% reduction
            - First-mile connectivity: +30% for outer boroughs
            - Public health: 2,500 avoided car trips daily
            
            Timeline: 5 years, 250 miles of protected lanes
            Cost: $500M capital
            ROI: Health + congestion savings = $1.2B over 10 years
            """,
        }
        
        return scenarios_output.get(scenario, f"Scenario: {scenario}\n[Analysis would go here]")
    
    def run_full_analysis(self) -> Dict[str, str]:
        """Run complete mobility analysis"""
        logger.info("Running full urban mobility analysis...")
        
        dataset = self.load_analysis_dataset()
        
        if not dataset:
            logger.error("No data available for analysis")
            return {}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "city": "New York City",
            "analyses": {}
        }
        
        # Congestion Analysis
        logger.info("Generating congestion analysis...")
        taxi_data = dataset.get("data_sources", {}).get("taxi", {})
        results["analyses"]["congestion"] = self.generate_congestion_summary(taxi_data)
        
        # Transit Accessibility
        logger.info("Analyzing transit accessibility...")
        gtfs_data = dataset.get("data_sources", {}).get("gtfs", {})
        results["analyses"]["transit_accessibility"] = self.generate_transit_accessibility_report(gtfs_data)
        
        # Scenario Analysis
        logger.info("Running scenario analyses...")
        scenarios = ["close_line_L", "reduce_bus_service", "add_bike_lanes"]
        for scenario in scenarios:
            results["analyses"][f"scenario_{scenario}"] = self.scenario_analysis(scenario)
        
        return results
    
    def save_results(self, results: Dict) -> Path:
        """Save analysis results to file"""
        output_file = RESULTS_DIR / f"mobility_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"✓ Results saved: {output_file}")
        
        # Also save readable report
        report_file = RESULTS_DIR / f"mobility_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w") as f:
            f.write("URBAN MOBILITY ASSESSMENT - NYC\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {results.get('timestamp')}\n\n")
            
            for analysis_name, content in results.get("analyses", {}).items():
                f.write(f"\n{'='*70}\n")
                f.write(f"{analysis_name.upper()}\n")
                f.write(f"{'='*70}\n\n")
                f.write(str(content) + "\n")
        
        logger.info(f"✓ Report saved: {report_file}")
        
        return output_file


def main():
    """Run LLM-based mobility analysis"""
    print("\n" + "="*60)
    print("Urban Mobility LLM Analyzer")
    print("="*60 + "\n")
    
    # Initialize analyzer
    analyzer = MobilityLLMAnalyzer(llm_model="gpt-4")
    
    # Run analysis
    results = analyzer.run_full_analysis()
    
    # Save results
    if results:
        analyzer.save_results(results)
        print("\n" + "="*60)
        print("✓ Analysis complete!")
        print(f"  Results: {RESULTS_DIR}")
        print("="*60)
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
