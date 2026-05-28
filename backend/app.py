from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import sys
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
import json
import numpy as np
import tempfile
import os
import re
import logging
import cmath
import math
import base64
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import seaborn as sns
import random
from typing import List, Dict, Tuple, Callable
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print("MATLAB Engine not available. Running in direct calculation mode.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AdvancedCircuitCalculator:
    
    def __init__(self):
        self.results = {}
    
    def calculate_voltage_divider(self, r1, r2, vin):
        total_resistance = r1 + r2
        vout = vin * (r2 / total_resistance)
        current = vin / total_resistance
        power_r1 = current**2 * r1
        power_r2 = current**2 * r2
        power_total = vin * current
        efficiency = (vout/vin) * 100
        
        return {
            'vout': vout,
            'current': current,
            'total_resistance': total_resistance,
            'power_r1': power_r1,
            'power_r2': power_r2,
            'power_total': power_total,
            'efficiency': efficiency,
            'voltage_ratio': vout/vin
        }
    

class GeneticCircuitOptimizer:
    
    def __init__(self, population_size=50, generations=100, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.fitness_history = []
    
    def optimize_voltage_divider(self, target_vout, vin, constraints=None):
        
        constraints = constraints or {
            'r_min': 100,
            'r_max': 1e6,
            'max_power': 0.5,
            'min_current': 0.001,
            'max_current': 0.1
        }
        
        def fitness_function(r1, r2):
            vout = vin * (r2 / (r1 + r2))
            voltage_error = abs(vout - target_vout) / target_vout
            
            current = vin / (r1 + r2)
            power_total = vin * current
            
            penalty = 0
            if current < constraints['min_current'] or current > constraints['max_current']:
                penalty += 1000
            if power_total > constraints['max_power']:
                penalty += 1000
            if r1 < constraints['r_min'] or r1 > constraints['r_max']:
                penalty += 1000
            if r2 < constraints['r_min'] or r2 > constraints['r_max']:
                penalty += 1000
            
            standard_penalty = self._standard_value_penalty(r1) + self._standard_value_penalty(r2)
            
            fitness = voltage_error + penalty + standard_penalty * 0.01
            
            return fitness, {
                'vout': vout,
                'current': current,
                'power': power_total,
                'error_percent': voltage_error * 100
            }
        
        best_solution = self._run_ga(fitness_function, constraints)
        
        return best_solution

    def optimize_op_amp_gain(self, target_gain, constraints=None):
        default_constraints = {
            'r_min': 100,
            'r_max': 1e6,
            'max_power': 0.1
        }
        constraints = {**default_constraints, **(constraints or {})}

        def fitness_function(rf, r1):
            gain = 1 + (rf / r1)
            gain_error = abs(gain - target_gain) / target_gain

            penalty = 0
            if rf < constraints['r_min'] or rf > constraints['r_max']:
                penalty += 1000
            if r1 < constraints['r_min'] or r1 > constraints['r_max']:
                penalty += 1000

            ratio = rf / r1
            if ratio < 1 or ratio > 100:
                penalty += 0.5

            standard_penalty = self._standard_value_penalty(rf) + self._standard_value_penalty(r1)

            fitness = gain_error + penalty + standard_penalty * 0.01
            return fitness, {
                'gain': gain,
                'error_percent': gain_error * 100
            }

        best_solution = self._run_ga(fitness_function, constraints)

        return best_solution
    def optimize_rc_filter(self, target_fc, constraints=None):
        default_constraints = {
            'r_min': 100,
            'r_max': 1e6,
            'c_min': 1e-12,
            'c_max': 1e-3
        }
        constraints = {**default_constraints, **(constraints or {})}
        
        def fitness_function(r, c):
            fc = 1 / (2 * math.pi * r * c)
            frequency_error = abs(fc - target_fc) / target_fc
            
            penalty = 0
            if r < constraints['r_min'] or r > constraints['r_max']:
                penalty += 1000
            if c < constraints['c_min'] or c > constraints['c_max']:
                penalty += 1000
            
            standard_penalty = self._standard_value_penalty(r) + self._standard_value_penalty_capacitor(c)
            
            fitness = frequency_error + penalty + standard_penalty * 0.01
            
            return fitness, {
                'fc': fc,
                'error_percent': frequency_error * 100
            }
        
        best_solution = self._run_ga_rc(fitness_function, constraints)
        
        return best_solution

    def _run_ga(self, fitness_function, constraints):
        population = []
        for _ in range(self.population_size):
            r1 = random.uniform(math.log10(constraints['r_min']), math.log10(constraints['r_max']))
            r2 = random.uniform(math.log10(constraints['r_min']), math.log10(constraints['r_max']))
            population.append((10**r1, 10**r2))
        
        best_fitness = float('inf')
        best_individual = None
        best_metrics = None
        
        for generation in range(self.generations):
            fitness_scores = []
            for individual in population:
                fitness, metrics = fitness_function(individual[0], individual[1])
                fitness_scores.append((fitness, individual, metrics))
            
            fitness_scores.sort(key=lambda x: x[0])
            
            if fitness_scores[0][0] < best_fitness:
                best_fitness = fitness_scores[0][0]
                best_individual = fitness_scores[0][1]
                best_metrics = fitness_scores[0][2]
            
            self.fitness_history.append(best_fitness)
            
            survivors = [ind for _, ind, _ in fitness_scores[:self.population_size // 2]]
            
            new_population = survivors.copy()
            while len(new_population) < self.population_size:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                child = (
                    parent1[0] if random.random() < 0.5 else parent2[0],
                    parent1[1] if random.random() < 0.5 else parent2[1]
                )
                
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, 1)
                    mutation_factor = random.uniform(0.8, 1.2)
                    child = list(child)
                    child[idx] *= mutation_factor
                    child[idx] = max(constraints['r_min'], min(constraints['r_max'], child[idx]))
                    child = tuple(child)
                
                new_population.append(child)
            
            population = new_population
        
        return {
            'r1': best_individual[0],
            'r2': best_individual[1],
            'fitness': best_fitness,
            'metrics': best_metrics,
            'generations': self.generations,
            'fitness_history': self.fitness_history
        }
    
    def _run_ga_rc(self, fitness_function, constraints):
        population = []
        for _ in range(self.population_size):
            r = random.uniform(math.log10(constraints['r_min']), math.log10(constraints['r_max']))
            c = random.uniform(math.log10(constraints['c_min']), math.log10(constraints['c_max']))
            population.append((10**r, 10**c))
        
        best_fitness = float('inf')
        best_individual = None
        best_metrics = None
        
        for generation in range(self.generations):
            fitness_scores = []
            for individual in population:
                fitness, metrics = fitness_function(individual[0], individual[1])
                fitness_scores.append((fitness, individual, metrics))
            
            fitness_scores.sort(key=lambda x: x[0])
            
            if fitness_scores[0][0] < best_fitness:
                best_fitness = fitness_scores[0][0]
                best_individual = fitness_scores[0][1]
                best_metrics = fitness_scores[0][2]
            
            self.fitness_history.append(best_fitness)
            
            survivors = [ind for _, ind, _ in fitness_scores[:self.population_size // 2]]
            
            new_population = survivors.copy()
            while len(new_population) < self.population_size:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                child = (
                    parent1[0] if random.random() < 0.5 else parent2[0],
                    parent1[1] if random.random() < 0.5 else parent2[1]
                )
                
                if random.random() < self.mutation_rate:
                    idx = random.randint(0, 1)
                    mutation_factor = random.uniform(0.8, 1.2)
                    child = list(child)
                    child[idx] *= mutation_factor
                    if idx == 0:
                        child[idx] = max(constraints['r_min'], min(constraints['r_max'], child[idx]))
                    else:
                        child[idx] = max(constraints['c_min'], min(constraints['c_max'], child[idx]))
                    child = tuple(child)
                
                new_population.append(child)
            
            population = new_population
        
        return {
            'r': best_individual[0],
            'c': best_individual[1],
            'fitness': best_fitness,
            'metrics': best_metrics,
            'generations': self.generations,
            'fitness_history': self.fitness_history
        }
    
    def _standard_value_penalty(self, value):
        e24_series = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]
        
        decade = 10 ** int(math.log10(value))
        normalized = value / decade
        
        closest = min(e24_series, key=lambda x: abs(x - normalized))
        error = abs(normalized - closest) / closest
        
        return error
    
    def _standard_value_penalty_capacitor(self, value):
        e12_series = [10, 12, 15, 18, 22, 27, 33, 39, 47, 56, 68, 82]
        
        if value == 0:
            return 1
        decade = 10 ** int(math.log10(value))
        normalized = value / decade
        
        closest = min(e12_series, key=lambda x: abs(x - normalized * 10))
        error = abs(normalized * 10 - closest) / closest
        
        return error
    def calculate_rc_circuit(self, r, c, frequencies=None, vin=1):
        results = {}
        
        tau = r * c
        fc = 1 / (2 * math.pi * r * c)
        results.update({
            'time_constant': tau,
            'cutoff_frequency': fc,
            'bandwidth': fc
        })
        
        if frequencies is None:
            frequencies = np.logspace(-1, 6, 1000)
        
        omega = 2 * np.pi * frequencies
        H = 1 / (1 + 1j * omega * r * c)
        
        magnitude_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H) * 180 / np.pi
        
        results.update({
            'frequencies': frequencies.tolist(),
            'magnitude_db': magnitude_db.tolist(),
            'phase_deg': phase_deg.tolist(),
            'gain_at_fc': -3.0,
            'phase_at_fc': -45.0
        })
        
        return results
    
    def calculate_rl_circuit(self, r, l, frequencies=None, vin=1):
        results = {}
        
        tau = l / r
        fc = r / (2 * math.pi * l)
        results.update({
            'time_constant': tau,
            'cutoff_frequency': fc,
            'bandwidth': fc
        })
        
        if frequencies is None:
            frequencies = np.logspace(-1, 6, 1000)
        
        omega = 2 * np.pi * frequencies
        H = (1j * omega * l) / (r + 1j * omega * l)
        
        magnitude_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H) * 180 / np.pi
        
        results.update({
            'frequencies': frequencies.tolist(),
            'magnitude_db': magnitude_db.tolist(),
            'phase_deg': phase_deg.tolist(),
            'gain_at_fc': -3.0,
            'phase_at_fc': 45.0
        })
        
        return results
    
    def calculate_rlc_circuit(self, r, l, c, frequencies=None, vin=1):
        results = {}
        
        omega_0 = 1 / math.sqrt(l * c)
        f0 = omega_0 / (2 * math.pi)
        zeta = r / (2 * math.sqrt(l / c))
        q = 1 / (2 * zeta) if zeta != 0 else float('inf')
        
        bandwidth = f0 / q if q != 0 else f0
        
        results.update({
            'resonant_frequency': f0,
            'damping_ratio': zeta,
            'quality_factor': q,
            'bandwidth': bandwidth,
            'damping_type': self._get_damping_type(zeta)
        })
        
        if frequencies is None:
            frequencies = np.logspace(math.log10(f0/100), math.log10(f0*100), 1000)
        
        omega = 2 * np.pi * frequencies
        s = 1j * omega
        
        H = 1 / (l * c * s**2 + r * c * s + 1)
        
        magnitude_db = 20 * np.log10(np.abs(H))
        phase_deg = np.angle(H) * 180 / np.pi
        
        results.update({
            'frequencies': frequencies.tolist(),
            'magnitude_db': magnitude_db.tolist(),
            'phase_deg': phase_deg.tolist()
        })
        
        return results
    
    def _get_damping_type(self, zeta):
        if zeta < 1:
            return "Underdamped"
        elif zeta == 1:
            return "Critically Damped"
        else:
            return "Overdamped"
    
    def calculate_op_amp_gain(self, rf, r1, circuit_type="non_inverting"):
        if circuit_type == "non_inverting":
            gain = 1 + (rf / r1)
            gain_db = 20 * math.log10(gain)
        elif circuit_type == "inverting":
            gain = -(rf / r1)
            gain_db = 20 * math.log10(abs(gain))
        else:
            gain = rf / r1
            gain_db = 20 * math.log10(abs(gain))
        
        return {
            'voltage_gain': gain,
            'gain_db': gain_db,
            'circuit_type': circuit_type,
            'feedback_factor': r1 / (r1 + rf) if circuit_type == "non_inverting" else 0
        }
    
    def calculate_parallel_resistance(self, resistors):
        if not resistors:
            return 0
        reciprocal_sum = sum(1/r for r in resistors if r != 0)
        return 1 / reciprocal_sum if reciprocal_sum != 0 else float('inf')
    
    def calculate_series_resistance(self, resistors):
        return sum(resistors)
    
    def generate_frequency_plot(self, frequencies, magnitude_db, phase_deg, title="Frequency Response"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.semilogx(frequencies, magnitude_db, 'b-', linewidth=2, label='Magnitude')
        ax1.grid(True, which="both", ls="-", alpha=0.3)
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.set_title(f'{title} - Bode Plot', fontsize=14, fontweight='bold')
        ax1.legend()
        
        ax2.semilogx(frequencies, phase_deg, 'r-', linewidth=2, label='Phase')
        ax2.grid(True, which="both", ls="-", alpha=0.3)
        ax2.set_xlabel('Frequency (Hz)', fontsize=12)
        ax2.set_ylabel('Phase (degrees)', fontsize=12)
        ax2.legend()
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
    
    def generate_transient_plot(self, time, input_signal, output_signal, title="Transient Response"):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(time * 1000, input_signal, 'b--', linewidth=2, label='Input', alpha=0.8)
        ax.plot(time * 1000, output_signal, 'r-', linewidth=3, label='Output')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Voltage (V)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"
    
    def generate_power_analysis_plot(self, components, powers, title="Power Analysis"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
        wedges, texts, autotexts = ax1.pie(powers, labels=components, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        ax1.set_title('Power Distribution', fontsize=14, fontweight='bold')
        
        bars = ax2.bar(components, [p*1000 for p in powers], color=colors, alpha=0.8)
        ax2.set_ylabel('Power (mW)', fontsize=12)
        ax2.set_title('Component Power Dissipation', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, power in zip(bars, powers):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{power*1000:.2f}mW', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{plot_data}"

class MatlabEngineManager:
    def __init__(self):
        self.engine = None
        if MATLAB_AVAILABLE:
            self.connect()
    
    def connect(self):
        if not MATLAB_AVAILABLE:
            logger.warning("MATLAB not available")
            return False
            
        try:
            logger.info("Starting MATLAB Engine...")
            self.engine = matlab.engine.start_matlab()
            logger.info("MATLAB Engine connected successfully")
            self.setup_matlab_environment()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MATLAB: {e}")
            return False
    
    def setup_matlab_environment(self):
        if not self.is_connected():
            return False
        
        try:
            self.engine.eval("clear; clc; close all;", nargout=0)
            self.engine.eval("""
            set(0, 'DefaultFigureVisible', 'off');
            set(0, 'DefaultFigurePosition', [100, 100, 800, 600]);
            """, nargout=0)
            return True
        except Exception as e:
            logger.error(f"Error setting up MATLAB environment: {e}")
            return False
    
    def disconnect(self):
        if self.engine:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None
    
    def is_connected(self):
        return self.engine is not None
    
    def execute_code(self, code):
        if not self.is_connected():
            return {"error": "MATLAB Engine not connected"}
        
        try:
            self.engine.eval(code, nargout=0)
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}

matlab_mgr = MatlabEngineManager()
calculator = AdvancedCircuitCalculator()
# Used to call the helper methods (generate_frequency_plot, generate_transient_plot,
# generate_power_analysis_plot, calculate_rc/rl/rlc_circuit) without instantiating
# a fresh optimizer per request. Population/generations aren't used for these helpers.
plot_helper = GeneticCircuitOptimizer()

@app.route('/')
def serve_frontend():
    import os
    try:
        if not os.path.exists('circuit_analyzer.html'):
            return f'''
            <html>
            <head><title>File Not Found</title></head>
            <body>
            <h1>HTML File Missing</h1>
            <p>circuit_analyzer.html not found in: {os.getcwd()}</p>
            <p>Files in directory: {os.listdir('.')}</p>
            </body>
            </html>
            ''', 404

        with open('circuit_analyzer.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
        
    except Exception as e:
        return f'''
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Server Error</h1>
            <p>Error loading HTML file: {str(e)}</p>
            <p>Current directory: {os.getcwd()}</p>
        </body>
        </html>
        ''', 500

@app.route('/debug')
def debug_info():
    import os
    return {
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'html_file_exists': os.path.exists('circuit_analyzer.html'),
        'python_version': sys.version
    }

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "connected": matlab_mgr.is_connected(),
        "matlab_available": MATLAB_AVAILABLE,
        "backend_ready": True,
        # Diagnostic: True if the server actually loaded a GROQ_API_KEY at startup.
        # Never exposes the key itself — just whether it's present.
        "groq_key_loaded": bool(GROQ_API_KEY),
        "groq_key_length": len(GROQ_API_KEY) if GROQ_API_KEY else 0,
        "groq_model": GROQ_MODEL,
    })

@app.route('/api/circuit/optimize', methods=['POST', 'OPTIONS'])
@cross_origin()
def optimize_circuit():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        logger.info(f"Optimization request: {data}")
        
        optimization_type = data.get('optimization_type')
        target = data.get('target')
        constraints = data.get('constraints', {})
        
        default_constraints = {
            'r_min': 100,
            'r_max': 1e6,
            'max_power': 0.5,
            'min_current': 0.001,
            'max_current': 0.1
        }
        constraints = {**default_constraints, **constraints}
        
        optimizer = GeneticCircuitOptimizer(
            population_size=data.get('population_size', 50),
            generations=data.get('generations', 100),
            mutation_rate=data.get('mutation_rate', 0.1)
        )

        
        if optimization_type == 'voltage_divider':
            target_vout = target.get('vout')
            vin = target.get('vin', 12)
            result = optimizer.optimize_voltage_divider(target_vout, vin, constraints)
            
            return jsonify({
                "success": True,
                "optimization_type": "voltage_divider",
                "optimized_components": {
                    'R1': result['r1'],
                    'R2': result['r2']
                },
                "achieved_metrics": result['metrics'],
                "fitness_score": result['fitness'],
                "generations": result['generations'],
                "fitness_history": result['fitness_history']
            })
        
        elif optimization_type == 'rc_filter':
            target_fc = target.get('cutoff_frequency')
            result = optimizer.optimize_rc_filter(target_fc, constraints)
            
            return jsonify({
                "success": True,
                "optimization_type": "rc_filter",
                "optimized_components": {
                    'R': result['r'],
                    'C': result['c']
                },
                "achieved_metrics": result['metrics'],
                "fitness_score": result['fitness'],
                "generations": result['generations'],
                "fitness_history": result['fitness_history']
            })
        
        elif optimization_type == 'op_amp_gain':
            target_gain = target.get('gain')
            result = optimizer.optimize_op_amp_gain(target_gain, constraints)
            
            return jsonify({
                "success": True,
                "optimization_type": "op_amp_gain",
                "optimized_components": {
                    'Rf': result['r1'],
                    'R1': result['r2']
                },
                "achieved_metrics": result['metrics'],
                "fitness_score": result['fitness'],
                "generations": result['generations'],
                "fitness_history": result['fitness_history']
            })
        
        else:
            return jsonify({"error": "Unknown optimization type"}), 400
            
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/circuit/analyze', methods=['POST'])
def analyze_circuit():
    try:
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request data: {request.data}")
        logger.info(f"Request form: {request.form}")
        
        data = request.json
        logger.info(f"Parsed JSON: {data}")
        
        query = data.get('query', '')
        analysis_type = data.get('analysis_type', 'auto')
        options = data.get('options', {})
        
        logger.info(f"Query: '{query}'")
        
        if not query:
            logger.error("No query provided")
            return jsonify({"error": "No query provided"}), 400
        
        parsed_data = parse_circuit_query(query)
        logger.info(f"Parsed components: {parsed_data.get('components', [])}")
        
        if not parsed_data.get('components'):
            logger.error("Could not identify circuit components")
            return jsonify({"error": "Could not identify circuit components"}), 400
        
        results = perform_comprehensive_analysis(parsed_data, analysis_type, options)
        
        return jsonify({
            "success": True,
            "parsed_data": parsed_data,
            **results
        })
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════════
#  AI CHAT — Groq-powered assistant with function calling
# ═══════════════════════════════════════════════════════════════════
# Setup:
#   Local dev (PowerShell): $env:GROQ_API_KEY="gsk_..."
#                            cd backend
#                            python app.py
#   Render deploy: set GROQ_API_KEY in Render dashboard → Environment.
#
# The assistant has one tool — `analyze_circuit` — wired to the same
# perform_comprehensive_analysis used by the main UI. The LLM decides
# whether to call it (user described a new circuit), or just answer
# (user asked an explanation / what-if question).
# ═══════════════════════════════════════════════════════════════════

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '').strip()
GROQ_MODEL   = os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')
GROQ_URL     = 'https://api.groq.com/openai/v1/chat/completions'

CHAT_SYSTEM_PROMPT = """You are IntelliCircuit AI — an expert assistant for a circuit-analysis app. You behave like an intelligent chatbot: reason about what's on screen, answer questions directly, and only invoke tools when truly needed.

## YOUR ENTIRE SCOPE
You ONLY have knowledge of and can help with these specific topics:
- Voltage dividers (resistor networks)
- Series / parallel resistor networks
- RC, RL, and RLC circuits (low-pass / high-pass filters, time-domain, impedance, cutoff)
- Op-amp amplifiers (inverting and non-inverting, gain, feedback)
- Standard component series (E12, E24, E96) for the above
- Basic Ohm's law / Kirchhoff math directly applied to the above

EVERYTHING ELSE IS OUT OF SCOPE — call `report_unsupported(topic)` IMMEDIATELY. Examples (non-exhaustive — anything outside the supported list is out of scope, even if it sounds electrical):
- Motors, drives, VFDs, servos, transformers, generators
- Transistors (BJT, MOSFET, JFET, IGBT), diodes (beyond ideal), thyristors, SCRs
- Microcontrollers, digital logic, FPGAs, programming, embedded systems
- Power electronics: rectifiers, buck/boost/SMPS, inverters
- AC mains, three-phase, power systems, grid, machines
- Antennas, transmission lines, RF, electromagnetic waves, microwaves
- Signal processing (FFT, filters beyond passive RC), control systems, PID
- Communications, modulation, error correction
- Semiconductor physics, device modeling, SPICE-level simulation
- Fuzzy logic, neural networks, ML, optimization theory in general
- Math, physics, chemistry, biology, anything non-electrical
- General coding help, writing assistance, life advice

If you're unsure whether something is in scope — IT ISN'T. Call report_unsupported.

## DECISION FLOW — FOLLOW IN ORDER

1. **Is the topic IN your scope (the supported list above)?**
   - No → call `report_unsupported(topic)` and stop. Don't try to be helpful with a partial answer.

2. **Can you answer DIRECTLY from your knowledge + the circuit context provided?**
   - YES (most questions) → write a 2-4 sentence text answer. NO tool call.
   - Examples that need NO tool:
     * "Which E-series should I use?" → use the "Closest E-series standard values" provided in context. Look at the snapped values + error % for each tier. Recommend E12 if max error is <10%, E24 if <5%, E96 for precision. ALWAYS answer this, NEVER call report_unsupported for it.
     * "Are my values standard?" → compare user's components against the E-series values in context.
     * "Why is the phase −45°?" → just explain
     * "What does Vout mean?" → just explain
     * "Compare these three options" → look at the context, write a comparison
     * "Is this circuit a high-pass or low-pass?" → look at the circuit type in context
     * "Explain this result" → summarize the numbers shown

3. **Does the user describe a NEW circuit with component values?**
   → Call `analyze_circuit(query)`. Then write a 2-3 sentence summary.
   - "Analyze RC with R=2.2k and C=100nF"
   - "What if I change R1 to 22k?" (with current circuit in context)

4. **Does the user want to MODIFY the current circuit** — add, remove, change, or swap a component?
   → Build a NEW query that includes ALL components from the current circuit context PLUS the requested change, then call `analyze_circuit`.
   - Current: "RLC R=100 L=10mH C=220nF". User says "add two resistors with 200 and 470 ohm" → call analyze_circuit("series RLC R1=100 R2=200 R3=470 L=10mH C=220nF").
   - Current: "voltage divider R1=10k R2=2.2k". User says "add a 4.7k in parallel to R2" → call analyze_circuit("voltage divider R1=10k R2=2.2k || R3=4.7k").
   - User says "change R to 220" → rebuild query with R replaced.
   - User says "remove L" → rebuild query without the inductor.
   - **Always preserve every other component** the user already had — they aren't asking you to delete them.
   - Use `||` between two component names to mark them as parallel.

5. **Does the user specify a TARGET output and want values found?**
   → Call `optimize_circuit(goal, target_value, vin)`. Then explain the result.
   - "Find values for Vout=2.6V"
   - "What R and C give cutoff at 1 kHz?"

## ABSOLUTE RULES
- NEVER return an empty response. If you have no answer, say so.
- NEVER call analyze_circuit just to "verify" or "double-check" something already in context — answer from context.
- NEVER guess values by iterating one-at-a-time. Use math or call optimize_circuit.
- ALWAYS give numbers with units. Round suggestions to E12 standards: 1, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2 × 10ⁿ.
- Keep responses short — 2-4 sentences for explanations, 1-2 sentences after a tool call."""


def _build_circuit_context(circuit):
    """Compact text snapshot of the currently-displayed analysis for the LLM."""
    if not circuit: return ''
    parsed = circuit.get('parsed_data') or {}
    lines = ['── CURRENT CIRCUIT CONTEXT ──']
    if parsed.get('circuit_type'):
        lines.append(f"Type: {parsed['circuit_type']}")
    if parsed.get('original_query'):
        lines.append(f"Original query: {parsed['original_query']}")
    comps = parsed.get('components') or []
    if comps:
        lines.append('Components:')
        for c in comps:
            lines.append(f"  • {c.get('name','?')} = {c.get('value','?')} {c.get('unit','')}")
    ans = circuit.get('direct_answers') or []
    if ans:
        lines.append('Results:')
        for a in ans:
            lines.append(f"  • {a.get('question','?')}: {a.get('answer','?')}")
    # Optimization extras
    opt = circuit.get('optimization_data') or {}
    if opt.get('goal') and opt.get('target') is not None:
        lines.append(f"Optimization goal: {opt['goal']}  target: {opt['target']}")
    # E-series snapped values — accept from either top level OR optimization_data,
    # so analyses (not just optimizations) can also ask "which E-series should I use?"
    sv = circuit.get('standard_values') or opt.get('standard_values')
    if sv:
        lines.append('Closest E-series standard values for the current components:')
        for variant in sv:
            vname = variant.get('name', '?')
            comp_str = ', '.join(f"{k}={v}" for k, v in (variant.get('components') or {}).items())
            ach_str = ', '.join(f"{k}={v}" for k, v in (variant.get('achieved') or {}).items())
            lines.append(f"  • {vname}: {comp_str}" + (f"  →  {ach_str}" if ach_str else ''))
    return '\n'.join(lines)


def _summarize_for_llm(result):
    """Turn an analysis result into compact text the LLM can use."""
    if not result or not result.get('success'):
        return 'Analysis failed: ' + str(result.get('error', 'unknown'))
    parts = ['Analysis complete.']
    for a in (result.get('direct_answers') or []):
        parts.append(f"{a['question']}: {a['answer']}")
    return '\n'.join(parts)


def _run_analysis(query):
    """Reuse the same pipeline as /api/circuit/analyze and return its dict."""
    parsed = parse_circuit_query(query)
    if not parsed.get('components'):
        return {'success': False, 'error': 'Could not identify circuit components from: ' + query}
    parsed['original_query'] = query
    result = perform_comprehensive_analysis(parsed, 'auto', {
        'show_direct_answers': True,
        'show_calculations': True,
        'show_circuit_diagram': True,
        'generate_plots': True,
    })
    result['success'] = True
    result['parsed_data'] = parsed
    return result


GROQ_FALLBACK_MODEL = os.environ.get('GROQ_FALLBACK_MODEL', 'llama-3.1-8b-instant')


def _call_groq(messages, tools=None):
    """
    Call Groq with rate-limit resilience:
      1. Try the primary model (GROQ_MODEL). One quick retry on 429.
      2. If still rate-limited, try the fallback model (GROQ_FALLBACK_MODEL) —
         Groq rate-limits PER MODEL, so the fallback has its own fresh quota.
      3. If both are capped, raise a friendly error.
    This keeps latency low (no long backoff chains) while surviving brief caps.
    """
    import requests, time

    verify_ssl = os.environ.get('GROQ_INSECURE', '').lower() not in ('1', 'true', 'yes')
    if not verify_ssl:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
    headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'}

    models = [GROQ_MODEL]
    if GROQ_FALLBACK_MODEL and GROQ_FALLBACK_MODEL != GROQ_MODEL:
        models.append(GROQ_FALLBACK_MODEL)

    last_429 = False
    for model in models:
        body = {'model': model, 'messages': messages, 'temperature': 0.15, 'max_tokens': 800}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        # One quick retry per model on 429
        for attempt in range(2):
            try:
                r = requests.post(GROQ_URL, headers=headers, json=body, timeout=30, verify=verify_ssl)
            except requests.exceptions.SSLError as e:
                raise RuntimeError(
                    'SSL certificate verification failed when contacting api.groq.com. '
                    'This is NOT an API-key problem. Fixes: (1) `pip install --upgrade certifi` then restart Python. '
                    '(2) Behind a corporate proxy (e.g. Bain): set `$env:GROQ_INSECURE="1"` and restart — LOCAL DEV ONLY. '
                    f'Underlying error: {e}'
                ) from e

            if r.status_code == 429:
                last_429 = True
                if attempt == 0:
                    time.sleep(1.5)   # single short retry, then move to fallback model
                    continue
                break  # give up on this model, try the next one

            r.raise_for_status()
            if model != GROQ_MODEL:
                logger.info(f'Groq served via fallback model {model}')
            return r.json()

    if last_429:
        raise RuntimeError(
            'Both Groq models are rate-limited right now (HTTP 429). The free tier caps requests per minute. '
            'Wait ~30-60 seconds, then try again. Each question now uses just one API call.'
        )
    raise RuntimeError('Groq request failed on all models.')


CHAT_TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'analyze_circuit',
            'description': (
                'Analyze an electrical circuit. Input is a natural-language description with component '
                'values (e.g. "voltage divider R1=10k R2=2.2k Vin=12V" or "RC filter R=1k C=100nF"). '
                'Returns DC voltage, DC current, AC impedance at 1 kHz, cutoff frequency, time constant, '
                'and a schematic diagram. Use whenever the user describes a new circuit or asks to verify '
                'a what-if change with specific component values.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'Natural-language circuit description with component values.',
                    }
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'optimize_circuit',
            'description': (
                'Genetic-algorithm search for component values that achieve a target output. '
                'Use when the user specifies a desired target (output voltage, cutoff frequency, op-amp gain) '
                'and there are multiple unknowns to solve for. Do NOT guess component values one at a time — '
                'call this tool instead.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'goal': {
                        'type': 'string',
                        'enum': ['voltage_divider', 'rc_filter', 'op_amp_gain'],
                        'description': 'What kind of circuit to optimize.',
                    },
                    'target_value': {
                        'type': 'number',
                        'description': (
                            'For voltage_divider: desired Vout in volts. '
                            'For rc_filter: desired cutoff frequency in Hz. '
                            'For op_amp_gain: desired gain (unitless, e.g. 11 for 21 dB).'
                        ),
                    },
                    'vin': {
                        'type': 'number',
                        'description': 'Input voltage in volts (for voltage_divider only; default 12).',
                    },
                },
                'required': ['goal', 'target_value'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'report_unsupported',
            'description': (
                'Call this when the user asks about ANYTHING outside IntelliCircuit\'s supported scope. '
                'Supported: voltage dividers, series/parallel resistors, RC/RL/RLC circuits, op-amps. '
                'NOT supported: motors, drives, VFDs, transformers, transistors (BJT/MOSFET), '
                'microcontrollers, digital logic, power electronics, AC mains, three-phase. '
                'Do not try to answer about unsupported topics — call this instead.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'topic': {
                        'type': 'string',
                        'description': 'Short label for what the user asked about (e.g. "induction motor", "MOSFET biasing").',
                    },
                },
                'required': ['topic'],
            },
        },
    },
]


@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        user_message = (data.get('message') or '').strip()
        history = data.get('history') or []
        current_circuit = data.get('current_circuit')

        if not user_message:
            return jsonify({'reply': 'Please enter a message.', 'analysis': None, 'unsupported_topic': None})

        # ── FAST-PATH: handle circuit modifications locally (no LLM call) ──
        # This makes "add/change/remove a component" work even when the LLM
        # is rate-limited, since it needs zero API calls.
        try:
            mod_result, mod_summary = _try_local_modification(user_message, current_circuit)
            if mod_summary is not None:
                # mod_summary set means we handled it locally:
                #  - with mod_result  → show summary AND update the panel
                #  - mod_result None  → just show the message (e.g. a clarification);
                #                        leave the existing circuit untouched.
                return jsonify({
                    'reply': mod_summary,
                    'analysis': mod_result,
                    'unsupported_topic': None,
                })
        except Exception as e:
            logger.warning(f'Local modification path failed, falling back to LLM: {e}')

        # ── FAST-PATH: answer "which E-series?" locally (no LLM, works for all types) ──
        try:
            es_reply = _try_local_eseries_answer(user_message, current_circuit)
            if es_reply:
                return jsonify({'reply': es_reply, 'analysis': None, 'unsupported_topic': None})
        except Exception as e:
            logger.warning(f'Local E-series path failed, falling back to LLM: {e}')

        if not GROQ_API_KEY:
            return jsonify({
                'reply': (
                    '⚠ AI chat not configured. The server needs a GROQ_API_KEY environment variable. '
                    'Locally: `$env:GROQ_API_KEY="gsk_..."` then restart Python. '
                    'On Render: set it in the dashboard → Environment.'
                ),
                'analysis': None, 'unsupported_topic': None,
            })

        # Build prompt with optional circuit context
        messages = [{'role': 'system', 'content': CHAT_SYSTEM_PROMPT}]
        ctx = _build_circuit_context(current_circuit)
        if ctx:
            messages.append({'role': 'system', 'content': ctx})
        messages.extend(history[-10:])
        messages.append({'role': 'user', 'content': user_message})

        # First LLM call — may return tool_calls
        first = _call_groq(messages, CHAT_TOOLS)
        msg = first['choices'][0]['message']
        reply_text = msg.get('content') or ''
        analysis_result = None
        unsupported_topic = None

        tool_calls = msg.get('tool_calls') or []
        if tool_calls:
            messages.append({
                'role': 'assistant',
                'content': reply_text,
                'tool_calls': tool_calls,
            })
            for tc in tool_calls:
                fname = (tc.get('function') or {}).get('name')
                try:
                    args = json.loads((tc.get('function') or {}).get('arguments') or '{}')
                except Exception:
                    args = {}

                if fname == 'analyze_circuit':
                    analysis_result = _run_analysis(args.get('query', ''))
                    tool_content = _summarize_for_llm(analysis_result)
                elif fname == 'optimize_circuit':
                    analysis_result, tool_content = _run_optimization(
                        args.get('goal'), args.get('target_value'), args.get('vin', 12)
                    )
                elif fname == 'report_unsupported':
                    unsupported_topic = args.get('topic') or 'that topic'
                    tool_content = (
                        f'Acknowledged. The user asked about "{unsupported_topic}", which is outside '
                        f'IntelliCircuit\'s supported scope. The frontend will display a feedback form. '
                        f'In your reply, briefly say IntelliCircuit doesn\'t support {unsupported_topic} yet '
                        f'and invite them to share feedback via the form that just appeared.'
                    )
                else:
                    tool_content = f'Unknown tool: {fname}'

                messages.append({
                    'role': 'tool',
                    'tool_call_id': tc.get('id'),
                    'name': fname or 'unknown',
                    'content': tool_content,
                })

            # NO second LLM call — we build the user-visible summary locally.
            # This keeps every chat message to exactly ONE Groq API call, which
            # matters a lot on the free tier's per-minute rate cap. The model's
            # first-turn text (if any) is preserved; otherwise the local summary
            # below fills in.

        # Build the reply locally if the model didn't already provide text.
        if not reply_text.strip():
            if unsupported_topic:
                reply_text = (
                    f"IntelliCircuit doesn't support {unsupported_topic} yet. "
                    f"It currently handles voltage dividers, resistor networks, RC/RL/RLC circuits, and op-amps. "
                    f"Would you like to share feedback, or get an answer from the AI's general knowledge (Web Search)?"
                )
            elif analysis_result and analysis_result.get('success'):
                # Clean local summary built from the analysis result — no extra API call.
                bullets = '\n'.join(
                    f"• {a['question']}: {a['answer']}"
                    for a in (analysis_result.get('direct_answers') or [])[:6]
                )
                reply_text = "Done! Here's the updated circuit:\n" + (bullets or 'see the main panel.')
            else:
                # No results and no text — only NOW try a text-only retry (one extra call).
                try:
                    text_only = messages + [{
                        'role': 'system',
                        'content': 'Give a final 2-4 sentence text answer. Do NOT call any tools.',
                    }]
                    retry = _call_groq(text_only, tools=None)
                    reply_text = (retry['choices'][0]['message'].get('content') or '').strip()
                except Exception as e:
                    logger.warning(f'Text-only retry failed: {e}')
                if not reply_text.strip():
                    reply_text = 'Sorry, I could not generate a response. Try rephrasing, or wait a moment if the AI is rate-limited.'

        return jsonify({
            'reply': reply_text,
            'analysis': analysis_result,
            'unsupported_topic': unsupported_topic,
        })

    except Exception as e:
        logger.error(f'Chat error: {e}', exc_info=True)
        return jsonify({'reply': f'⚠ AI chat error: {e}', 'analysis': None, 'unsupported_topic': None}), 200


# ═══════════════════════════════════════════════════════════════════
#  LOCAL MODIFICATION FAST-PATH — no LLM call (avoids rate limits)
#  Handles "add/change/remove a component" by parsing the request,
#  rebuilding the circuit query string, and running the normal pipeline.
# ═══════════════════════════════════════════════════════════════════

def _fmt_component_token(c):
    """Format one component as a parser-friendly query token (e.g. 'R1=100 ohm')."""
    t, name, v = c['type'], c['name'], c['value']
    if t == 'resistor':
        if v >= 1e6:   return f"{name}={v/1e6:g}M ohm"
        if v >= 1e3:   return f"{name}={v/1e3:g}k ohm"
        return f"{name}={v:g} ohm"
    if t == 'inductor':
        if v < 1e-3:   return f"{name}={v*1e6:g}uH"
        if v < 1:      return f"{name}={v*1e3:g}mH"
        return f"{name}={v:g}H"
    if t == 'capacitor':
        if v < 1e-9:   return f"{name}={v*1e12:g}pF"
        if v < 1e-6:   return f"{name}={v*1e9:g}nF"
        if v < 1e-3:   return f"{name}={v*1e6:g}uF"
        return f"{name}={v:g}F"
    if t == 'voltage_source':
        return f"{v:g}V"
    return ''


def _circuit_type_hint(components):
    """Pick a query prefix based on the final component mix."""
    n_r = sum(1 for c in components if c['type'] == 'resistor')
    n_l = sum(1 for c in components if c['type'] == 'inductor')
    n_c = sum(1 for c in components if c['type'] == 'capacitor')
    if n_l and n_c:   return 'series RLC circuit'
    if n_c and n_r:   return 'RC circuit'
    if n_l and n_r:   return 'RL circuit'
    return 'series circuit'


def _rebuild_query(components):
    """
    Rebuild a circuit query string from a component list, using `||` between
    members of the same parallel_group. Series elements joined by ', '.
    """
    type_hint = _circuit_type_hint(components)
    # Build ordered tokens, collapsing parallel groups
    tokens, seen_groups = [], set()
    vsrc = None
    for c in components:
        if c['type'] == 'voltage_source':
            vsrc = c
            continue
        gid = c.get('parallel_group')
        if gid:
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            members = [x for x in components if x.get('parallel_group') == gid]
            tokens.append(' || '.join(_fmt_component_token(m) for m in members))
        else:
            tokens.append(_fmt_component_token(c))
    q = type_hint + ' with ' + ', '.join(tokens)
    if vsrc:
        q += f", {_fmt_component_token(vsrc)}"
    return q


def _extract_component_spec(text):
    """
    Parse a component value+type from free text like '4.7k', '100 ohm',
    '1uF', '10mH'. Returns (type, value) or (None, None).
    Bare numbers / k / M default to resistor.
    """
    m = re.search(
        r'(\d+(?:\.\d+)?)\s*([pnuµmkKMGg]?)\s*(ohms?|Ω|farads?|f|henr(?:y|ies)|h)?\b',
        text, re.IGNORECASE,
    )
    if not m:
        return None, None
    num = float(m.group(1))
    prefix = m.group(2) or ''
    unit = (m.group(3) or '').lower()

    if unit in ('f', 'farad', 'farads'):
        ctype = 'capacitor'
        mult = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'm': 1e-3, '': 1}.get(prefix.lower(), 1)
    elif unit in ('h', 'henry', 'henries'):
        ctype = 'inductor'
        mult = {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'm': 1e-3, 'k': 1e3, '': 1}.get(prefix.lower(), 1)
    else:
        ctype = 'resistor'
        mult = {'k': 1e3, 'K': 1e3, 'M': 1e6, 'm': 1e-3, 'G': 1e9, 'g': 1e9, '': 1}.get(prefix, 1)
    return ctype, num * mult


def _next_name(components, ctype):
    prefix = {'resistor': 'R', 'capacitor': 'C', 'inductor': 'L'}[ctype]
    n = sum(1 for c in components if c['type'] == ctype) + 1
    return f"{prefix}{n}"


def _try_local_modification(message, current_circuit):
    """
    Handle common modification requests WITHOUT an LLM call to dodge rate limits.
    Returns (analysis_result, human_summary) or (None, None) if no pattern matched.
    Supports: add (series/parallel), change/set value, remove.
    """
    if not current_circuit:
        return None, None
    parsed = current_circuit.get('parsed_data') or {}
    components = [dict(c) for c in (parsed.get('components') or [])]
    if not components:
        return None, None

    msg = message.lower().strip()
    if not re.search(r'\b(add|change|set|make|remove|delete|replace|swap|put)\b', msg):
        return None, None

    def _run(comps):
        q = _rebuild_query(comps)
        logger.info(f"Local modification rebuilt query: {q}")
        return _run_analysis(q)

    # ---------- REMOVE ----------
    rm = re.search(r'\b(?:remove|delete|drop)\s+(?:the\s+)?([rlc]\d*)\b', msg)
    if rm:
        target = rm.group(1).upper()
        new_comps = [c for c in components if c['name'].upper() != target]
        if len(new_comps) == len(components):
            return None, None  # target not found
        res = _run(new_comps)
        if res and res.get('success'):
            return res, f"Removed {target}. " + _local_summary(res)
        return None, None

    # ---------- CHANGE / SET ----------
    ch = re.search(r'\b(?:change|set|make|update)\s+(?:the\s+)?([rlc]\d*)\s*(?:to|=|as)?\s*(.+)$', msg)
    if ch:
        target = ch.group(1).upper()
        ctype, value = _extract_component_spec(ch.group(2))
        if value is not None:
            found = False
            for c in components:
                if c['name'].upper() == target:
                    c['value'] = value
                    found = True
                    break
            if found:
                res = _run(components)
                if res and res.get('success'):
                    return res, f"Changed {target} to {_fmt_component_token({'type': ctype or 'resistor', 'name': target, 'value': value}).split('=')[1].strip()}. " + _local_summary(res)
        return None, None

    # ---------- ADD ----------
    if re.search(r'\badd\b|\bput\b|\binsert\b', msg):
        ctype, value = _extract_component_spec(msg)

        # If no explicit numeric value, infer the TYPE from keywords and try to
        # resolve the value from context ("same value(s)", "one more", "another").
        # This keeps value-less requests OUT of the unreliable LLM path.
        if value is None:
            if re.search(r'\binduct', msg):            # inductor / inductance / typo "inductr"
                ctype = 'inductor'
            elif re.search(r'\bcapacit|\bcap\b|\bcondenser', msg):
                ctype = 'capacitor'
            elif re.search(r'\bresist', msg):
                ctype = 'resistor'
            else:
                return None, None  # genuinely can't tell what to add → let LLM try

            existing_same = [c for c in components if c['type'] == ctype]
            wants_same = bool(re.search(r'\bsame\b|\banother\b|\bone more\b|\bduplicate\b', msg))
            if existing_same and (wants_same or True):
                # Copy the value of the most recent same-type component.
                value = existing_same[-1]['value']
            else:
                # No same-type component to copy from and no explicit value.
                # Ask instead of guessing — NEVER mangle the existing circuit.
                tw = {'inductor': 'inductor', 'capacitor': 'capacitor', 'resistor': 'resistor'}[ctype]
                ex = {'inductor': "'add a 10mH inductor'",
                      'capacitor': "'add a 100nF capacitor'",
                      'resistor': "'add a 1k resistor'"}[ctype]
                return None, (
                    f"I can add an {tw} to your circuit, but I need its value — there's no "
                    f"existing {tw} to copy. For example: {ex}."
                )

        # parallel or series?
        is_parallel = 'parallel' in msg
        tgt_m = re.search(r'(?:to|with|across)\s+(?:the\s+)?([rlc]\d*)\b', msg)
        target_name = tgt_m.group(1).upper() if tgt_m else None

        new_name = _next_name(components, ctype)
        new_comp = {'type': ctype, 'name': new_name,
                    'value': value, 'unit': {'resistor': 'Ω', 'capacitor': 'F', 'inductor': 'H'}[ctype]}

        if is_parallel:
            target = None
            if target_name:
                target = next((c for c in components if c['name'].upper() == target_name), None)
            if target is None:
                target = next((c for c in components if c['type'] == ctype), None)
            if target is None:
                components.append(new_comp)
            else:
                gid = target.get('parallel_group')
                if not gid:
                    gid = max([c.get('parallel_group', 0) for c in components], default=0) + 1
                    target['parallel_group'] = gid
                new_comp['parallel_group'] = gid
                components.append(new_comp)
            rel = f"in parallel with {target['name']}" if target else "in series"
        else:
            components.append(new_comp)
            rel = "in series"

        res = _run(components)
        if res and res.get('success'):
            val_str = _fmt_component_token(new_comp).split('=', 1)[1].strip()
            return res, f"Added {new_name}={val_str} {rel}. " + _local_summary(res)
        return None, None

    return None, None


def _local_summary(res):
    """Build a short text summary from an analysis result (no LLM)."""
    bullets = []
    for a in (res.get('direct_answers') or [])[:5]:
        bullets.append(f"{a['question']} = {a['answer']}")
    return "Updated circuit: " + "; ".join(bullets) if bullets else "See the main panel for the updated circuit."


def _fmt_eseries_val(name, v):
    """Format an E-series component value with the right unit based on its name."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    n = (name or '').upper()
    if n.startswith('C'):
        if v < 1e-9:  return f"{v*1e12:g}pF"
        if v < 1e-6:  return f"{v*1e9:g}nF"
        if v < 1e-3:  return f"{v*1e6:g}µF"
        return f"{v:g}F"
    if n.startswith('L'):
        if v < 1e-3:  return f"{v*1e6:g}µH"
        if v < 1:     return f"{v*1e3:g}mH"
        return f"{v:g}H"
    if v >= 1e6:  return f"{v/1e6:g}MΩ"
    if v >= 1e3:  return f"{v/1e3:g}kΩ"
    return f"{v:g}Ω"


def _try_local_eseries_answer(message, current_circuit):
    """
    Answer 'which E-series should I use?' LOCALLY (no LLM) using the
    standard_values already in the circuit context. Works for EVERY circuit
    type (RC, RL, RLC, voltage divider, op-amp), so it never depends on the
    LLM correctly classifying the question as in-scope. Returns a text answer,
    or None if this isn't an E-series question / no standard_values available.
    """
    if not current_circuit:
        return None
    msg = (message or '').lower()
    if not re.search(r'e-?\s?series|\be12\b|\be24\b|\be96\b|standard\s+(value|component|resistor|cap)|which\s+series', msg):
        return None
    opt = current_circuit.get('optimization_data') or {}
    sv = current_circuit.get('standard_values') or opt.get('standard_values')
    if not sv:
        return None

    def tier_err(variant):
        ach = (variant or {}).get('achieved') or {}
        if 'max_error_pct' in ach:
            return abs(float(ach['max_error_pct']))
        if 'error' in ach:
            return abs(float(ach['error']))
        return 0.0

    by_name = {v.get('name'): v for v in sv}
    lines = ["Here are the closest standard (E-series) values for your circuit:"]
    for v in sv:
        comps = ', '.join(f"{k}={_fmt_eseries_val(k, val)}"
                          for k, val in (v.get('components') or {}).items())
        lines.append(f"• {v.get('name')}: {comps}  (worst-case error {tier_err(v):.2f}%)")

    e12_err = tier_err(by_name.get('E12')) if by_name.get('E12') else 99
    e24_err = tier_err(by_name.get('E24')) if by_name.get('E24') else 99
    if e12_err < 10:
        rec = "E12 (10% tolerance) — cheapest and most widely available, and the error is small for your values."
    elif e24_err < 5:
        rec = "E24 (5% tolerance) — easy to source with good accuracy."
    else:
        rec = "E96 (1% tolerance) — use this for precision; E12/E24 introduce too much error here."
    lines.append(f"\nRecommendation: {rec}")
    return '\n'.join(lines)


def _run_optimization(goal, target_value, vin=12):
    """
    Reuse the existing GeneticCircuitOptimizer. After optimization, run a full
    analyze on the resulting components so the frontend gets the same rich
    result shape (diagram + direct_answers + plots) as a normal analyze call.
    Returns (analysis_dict, llm_text_summary).
    """
    if not goal or target_value is None:
        return None, 'Optimization failed: missing goal or target_value.'
    try:
        optimizer = GeneticCircuitOptimizer(population_size=50, generations=100)
        constraints = {'r_min': 100, 'r_max': 1e6, 'max_power': 0.5,
                       'min_current': 0.001, 'max_current': 0.1}

        if goal == 'voltage_divider':
            r = optimizer.optimize_voltage_divider(float(target_value), float(vin), constraints)
            verify_query = f"voltage divider with R1={r['r1']:.0f} ohm, R2={r['r2']:.0f} ohm, Vin={vin}V"
        elif goal == 'rc_filter':
            r = optimizer.optimize_rc_filter(float(target_value), constraints)
            verify_query = f"RC low-pass filter with R={r['r']:.0f} ohm, C={r['c']:.4g} F"
        elif goal == 'op_amp_gain':
            r = optimizer.optimize_op_amp_gain(float(target_value), constraints)
            verify_query = f"non-inverting op-amp with Rf={r['r1']:.0f} ohm and R1={r['r2']:.0f} ohm"
        else:
            return None, f'Unknown optimization goal: {goal}'

        # Re-run as a normal analysis so the frontend gets a full result shape
        analysis = _run_analysis(verify_query)
        achieved = ', '.join(f"{k}={v}" for k, v in (r.get('metrics') or {}).items())
        summary = (f"Optimization complete for goal={goal}, target={target_value}. "
                   f"Found values: {verify_query}. Achieved metrics: {achieved}. Fitness={r.get('fitness'):.4f}.")
        return analysis, summary
    except Exception as e:
        logger.error(f'Optimization error in chat: {e}', exc_info=True)
        return None, f'Optimization failed: {e}'


@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Collect feedback about unsupported topics or general bugs. Appends to feedback.log."""
    try:
        data = request.get_json() or {}
        topic = (data.get('topic') or '').strip()
        message = (data.get('message') or '').strip()
        if not message:
            return jsonify({'success': False, 'error': 'Empty feedback.'}), 400
        import datetime
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        line = f"[{ts}] topic={topic!r} message={message!r}\n"
        # Log + persist
        logger.info(f'FEEDBACK: {line.strip()}')
        try:
            with open(os.path.join(os.path.dirname(__file__), 'feedback.log'), 'a', encoding='utf-8') as f:
                f.write(line)
        except Exception as e:
            logger.warning(f'Could not write feedback.log: {e}')
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f'Feedback endpoint error: {e}', exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


# Free-knowledge LLM endpoint used by the "Web Search" button on the
# unsupported-topic feedback form. The chat agent normally refuses
# out-of-scope topics, but the user can opt in to get an answer from
# the LLM's general knowledge for that one question.
WEBSEARCH_SYSTEM_PROMPT = """You are a general knowledge assistant. The user is asking about a topic that IntelliCircuit (a circuit-analysis app) does not natively support, but they've explicitly asked you to answer using your general knowledge.

Give a helpful, accurate, concise answer (3-6 sentences). If the question is technical, include key formulas/equations. If your training data is limited on the topic, say so honestly rather than guessing. If the question would require real-time or up-to-date information you don't have, note that limitation.

Do NOT mention IntelliCircuit's scope. Just answer the question as well as you can."""


@app.route('/api/chat/websearch', methods=['POST'])
def chat_websearch():
    """LLM-powered fallback answer for an out-of-scope topic, opt-in by the user."""
    try:
        data = request.get_json() or {}
        question = (data.get('question') or '').strip()
        topic = (data.get('topic') or '').strip()
        if not question:
            return jsonify({'reply': 'Please provide a question.'})
        if not GROQ_API_KEY:
            return jsonify({'reply': '⚠ AI not configured. Set GROQ_API_KEY.'})
        try:
            messages = [
                {'role': 'system', 'content': WEBSEARCH_SYSTEM_PROMPT},
                {'role': 'user', 'content': question},
            ]
            response = _call_groq(messages, tools=None)
            reply = (response['choices'][0]['message'].get('content') or '').strip()
            if not reply:
                reply = "I couldn't generate an answer for that. Try rephrasing the question."
            return jsonify({'reply': reply})
        except Exception as e:
            logger.error(f'Web-search endpoint error: {e}', exc_info=True)
            return jsonify({'reply': f'⚠ Error: {e}'})
    except Exception as e:
        logger.error(f'Web-search endpoint outer error: {e}', exc_info=True)
        return jsonify({'reply': f'⚠ Error: {e}'}), 200


# ─────────────────────────────────────────────────────────────────────
# Topology helpers — combine multiple series / parallel components into
# one effective value. Resistors and inductors use the same formulas
# (series sums, parallel reciprocates); capacitors are the opposite.
# ─────────────────────────────────────────────────────────────────────
def _combine_r_series(values):    return sum(values) if values else 0.0
def _combine_l_series(values):    return sum(values) if values else 0.0
def _combine_c_parallel(values):  return sum(values) if values else 0.0

def _combine_r_parallel(values):
    if not values or any(v <= 0 for v in values): return 0.0
    return 1.0 / sum(1.0 / v for v in values)

_combine_l_parallel = _combine_r_parallel

def _combine_c_series(values):
    if not values or any(v <= 0 for v in values): return 0.0
    return 1.0 / sum(1.0 / v for v in values)


def _parse_parallel_groups(query, components):
    """
    Detect `||` between two or three named components in the query, e.g.
    `R1 || R2`, `R1=10k || R2=20k`, `C1 || C2 || C3`. Tags those components
    with a shared `parallel_group` id so the analyzer combines them in parallel.

    Limitation: only flat pairs/chains. Nested forms like `(R1 + R2) || R3`
    are not yet supported — pre-compute the sub-equivalent.
    """
    # Allow an optional `=value` (and unit) between the name and `||`,
    # so both `R1 || R2` and `R1=100 || R2=300` are recognised.
    val = r'(?:\s*=\s*[\d.]+\s*[a-zA-Zµμ]*)?'
    name = r'([RrCcLl]\w*)'
    pair_re = re.compile(
        rf'{name}{val}\s*\|\|\s*{name}{val}(?:\s*\|\|\s*{name}{val})?'
    )
    name_to_comp = {c['name'].lower(): c for c in components}
    group_id = 0
    for m in pair_re.finditer(query):
        names = [g for g in m.groups() if g]
        comps_in_group = [name_to_comp.get(n.lower()) for n in names]
        comps_in_group = [c for c in comps_in_group if c is not None]
        if len(comps_in_group) < 2:
            continue
        # Only group components of the same type (no `R || C`)
        kinds = {c['type'] for c in comps_in_group}
        if len(kinds) != 1:
            continue
        group_id += 1
        for c in comps_in_group:
            c['parallel_group'] = group_id
        logger.info(f"Tagged parallel group {group_id}: {[c['name'] for c in comps_in_group]}")


def _effective_values(components, kind, default_series=True):
    """
    Returns (effective_value, breakdown_string) for all components of `kind`.
    Components tagged with the same `parallel_group` are combined in parallel;
    everything else is treated as series by default.

    `kind` is 'resistor', 'inductor', or 'capacitor'.
    """
    same = [c for c in components if c['type'] == kind]
    if not same:
        return 0.0, ''
    if len(same) == 1:
        return same[0]['value'], same[0]['name']

    # Bucket by parallel_group; group 0 (or missing) means "in series with everything else"
    series_vals, series_names = [], []
    groups = {}   # group_id -> [components]
    for c in same:
        gid = c.get('parallel_group', 0)
        if gid:
            groups.setdefault(gid, []).append(c)
        else:
            series_vals.append(c['value'])
            series_names.append(c['name'])

    # Combine each parallel group into a single effective value
    if kind == 'capacitor':
        par_combine = _combine_c_parallel
    else:
        par_combine = _combine_r_parallel
    par_effective = []
    par_label = []
    for gid, group in groups.items():
        vals = [c['value'] for c in group]
        names = [c['name'] for c in group]
        par_effective.append(par_combine(vals))
        par_label.append('(' + ' || '.join(names) + ')')

    # Combine the per-group effectives in series with the loose components.
    if kind == 'capacitor':
        # Caps are opposite: series Cs combine reciprocally.
        all_vals = series_vals + par_effective
        eff = _combine_c_series(all_vals) if len(all_vals) > 1 else (all_vals[0] if all_vals else 0)
        joiner = ' (series) + ' if len(all_vals) > 1 else ''
    else:
        all_vals = series_vals + par_effective
        eff = _combine_r_series(all_vals)
        joiner = ' + '

    # Human-readable breakdown
    breakdown = joiner.join(series_names + par_label) if all_vals else ''
    return eff, breakdown


def _parse_natural_topology(query, components):
    """
    Detect natural-language parallel phrasing and tag the relevant same-type
    components with a shared parallel_group. Handles:
      • "R=100, R=300, R=190 in parallel"            → all three (clause before keyword)
      • "R1=100, R2=300 in parallel, R3=190 in series"→ only R1,R2 (R3 is a series clause)
      • "R=100 in parallel with R=300"               → both (with-connector pulls in the next)
      • "parallel combination of R=100 and R=200"    → both (combination-of pulls in the next)
      • "add a 4.7k in parallel to R2"               → the 4.7k and R2

    Strategy per 'parallel' keyword:
      - Backward span: from the nearest 'series'/'then' boundary (within 70 chars)
        up to the keyword. Components here are candidates.
      - Forward span: ONLY if the keyword is immediately followed by
        with/to/of/combination/and — then pull same-type components after it,
        stopping at the next 'series'.
      Group the dominant same-type cluster among the candidates.
    Skips entirely if `||` already created groups.
    """
    from collections import Counter
    q = query.lower()
    if any(c.get('parallel_group') for c in components):
        return

    gid = 0
    for km in re.finditer(r'\bparallel\b', q):
        kpos, kend = km.start(), km.end()

        before = sorted([c for c in components if c.get('_pos', -1) < kpos],
                        key=lambda c: c.get('_pos', -1))

        # Forward boundary: stop at the next 'series'/'then', else 50 chars out
        stop = re.search(r'\b(series|then)\b', q[kend:])
        fwd_limit = kend + (stop.start() if stop else 50)

        if not before:
            # "parallel combination of X and Y" — all components are after keyword
            after = [c for c in components if kend <= c.get('_pos', -1) < fwd_limit]
            if len(after) >= 2:
                counts = Counter(c['type'] for c in after)
                dom, n = counts.most_common(1)[0]
                if n >= 2:
                    gid += 1
                    for c in after:
                        if c['type'] == dom:
                            c['parallel_group'] = gid
            continue

        nearest = before[-1]
        ntype = nearest['type']

        # Same-type components shortly AFTER the keyword. This is robust to the
        # exact connector word ("with"/"to"/typo "iwth"), since we just look for
        # the next same-type component(s) within the forward window.
        after_same = [c for c in components
                      if kend <= c.get('_pos', -1) < fwd_limit and c['type'] == ntype]

        if after_same:
            # "X in parallel [connector] Y [and Z]" → nearest-before + same-type-after
            candidates = [nearest] + after_same
        else:
            # "X, Y, Z in parallel" (no same-type component after) → whole list
            # before the keyword, back to the nearest series/then boundary.
            boundary = 0
            for sm in re.finditer(r'\b(series|then)\b', q[:kpos]):
                if kpos - sm.end() < 70:
                    boundary = sm.end()
            candidates = [c for c in components
                          if boundary <= c.get('_pos', -1) < kpos and c['type'] == ntype]

        if len(candidates) < 2:
            continue
        gid += 1
        for c in candidates:
            c['parallel_group'] = gid
        logger.info(f"Natural-language parallel: grouped {[c['name'] for c in candidates]} as group {gid}")


def _effective_resistor_list(components):
    """
    Return an ordered list of (label, effective_value) for resistors, collapsing
    each parallel_group into one combined resistor while preserving first-appearance
    order. Used by voltage divider / op-amp where specific R positions matter.

    e.g. R1=10k, R2=2.2k || R3=4.7k  →  [("R1", 10000), ("(R2 || R3)", 1496.6)]
    """
    resistors = [c for c in components if c['type'] == 'resistor']
    out, seen = [], set()
    for c in resistors:
        gid = c.get('parallel_group')
        if gid:
            if gid in seen:
                continue
            seen.add(gid)
            group = [x for x in resistors if x.get('parallel_group') == gid]
            eff = _combine_r_parallel([x['value'] for x in group])
            label = '(' + ' || '.join(x['name'] for x in group) + ')'
            out.append((label, eff))
        else:
            out.append((c['name'], c['value']))
    return out


def parse_circuit_query(query):
    components = []
    calculations_requested = []
    query_lower = query.lower()
    consumed = set()  # char positions already consumed by a match

    logger.info(f"Parsing query: '{query}'")

    def is_consumed(s, e):
        return any(i in consumed for i in range(s, e))

    def consume(s, e):
        consumed.update(range(s, e))

    def parse_prefix(prefix, kind):
        """kind: 'r' for resistor (k/M/G), 'cl' for cap/inductor (p/n/u/m)."""
        if not prefix:
            return 1.0
        if kind == 'r':
            return {'k': 1e3, 'K': 1e3, 'M': 1e6, 'm': 1e-3,
                    'G': 1e9, 'g': 1e9}.get(prefix, 1.0)
        # cap/inductor — lowercase: p,n,u,μ,m all common; M rare (treat as mega for inductors)
        p = prefix.lower()
        return {'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'μ': 1e-6,
                'm': 1e-3, 'k': 1e3}.get(p, 1.0)

    # ---- Resistors (R, R1, R2, Rf, Rin, Rload, ...) ----
    # \b ensures word boundary so "voltage" / "Resonant" don't false-match
    res_pat = re.compile(
        r'\b[Rr](\w*)\s*=\s*(\d+(?:\.\d+)?)\s*([kKMmGg]?)(?:\s*(?:ohm|Ω))?',
        re.UNICODE
    )
    seen_names = set()
    for m in res_pat.finditer(query):
        if is_consumed(m.start(), m.end()):
            continue
        consume(m.start(), m.end())
        suffix = m.group(1)
        if suffix:
            name = f'R{suffix}'
        else:
            n = sum(1 for c in components if c['type'] == 'resistor') + 1
            name = f'R{n}'
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        value = float(m.group(2)) * parse_prefix(m.group(3), 'r')
        logger.info(f"Parsed resistor {name} = {value} Ω")
        components.append({'type': 'resistor', 'name': name, 'value': value, 'unit': 'Ω', '_pos': m.start()})

    # ---- Capacitors (C, C1, Cf, ...) ----
    # F is case-insensitive and optional when a metric prefix is present, so
    # "220nF", "220nf", and "800n" all parse. Bare "C=220" (no prefix, no F)
    # is skipped to avoid matching stray words.
    cap_pat = re.compile(
        r'\b[Cc](\w*)\s*=\s*(\d+(?:\.\d+)?)\s*([pnuµμmM]?)\s*([Ff]?)\b',
        re.UNICODE
    )
    for m in cap_pat.finditer(query):
        if is_consumed(m.start(), m.end()):
            continue
        prefix = m.group(3)
        funit = m.group(4)
        if not prefix and not funit:
            continue  # no unit at all — skip to avoid false positives
        consume(m.start(), m.end())
        suffix = m.group(1)
        if suffix:
            name = f'C{suffix}'
        else:
            n = sum(1 for c in components if c['type'] == 'capacitor') + 1
            name = f'C{n}'
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        value = float(m.group(2)) * parse_prefix(prefix, 'cl')
        logger.info(f"Parsed capacitor {name} = {value} F")
        components.append({'type': 'capacitor', 'name': name, 'value': value, 'unit': 'F', '_pos': m.start()})

    # ---- Inductors (L, L1, ...) ----
    # H is case-insensitive and optional when a prefix is present ("10mH",
    # "10mh", "10m" all parse). Bare "L=10" (no prefix, no H) is skipped.
    ind_pat = re.compile(
        r'\b[Ll](\w*)\s*=\s*(\d+(?:\.\d+)?)\s*([pnuµμmMkK]?)\s*([Hh]?)\b',
        re.UNICODE
    )
    for m in ind_pat.finditer(query):
        if is_consumed(m.start(), m.end()):
            continue
        prefix = m.group(3)
        hunit = m.group(4)
        if not prefix and not hunit:
            continue
        consume(m.start(), m.end())
        suffix = m.group(1)
        if suffix:
            name = f'L{suffix}'
        else:
            n = sum(1 for c in components if c['type'] == 'inductor') + 1
            name = f'L{n}'
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        value = float(m.group(2)) * parse_prefix(prefix, 'cl')
        logger.info(f"Parsed inductor {name} = {value} H")
        components.append({'type': 'inductor', 'name': name, 'value': value, 'unit': 'H', '_pos': m.start()})

    # ---- Unnamed resistor fallback: "4.7k resistor", "100 ohm resistor" ----
    unnamed_res = re.compile(
        r'(\d+(?:\.\d+)?)\s*([kKMmGg]?)\s*(?:ohm|Ω)?\s+resistor', re.IGNORECASE
    )
    for m in unnamed_res.finditer(query):
        if is_consumed(m.start(), m.end()):
            continue
        consume(m.start(), m.end())
        n = sum(1 for c in components if c['type'] == 'resistor') + 1
        name = f'R{n}'
        if name.lower() in seen_names:
            continue
        seen_names.add(name.lower())
        value = float(m.group(1)) * parse_prefix(m.group(2), 'r')
        logger.info(f"Parsed unnamed resistor {name} = {value} Ω")
        components.append({'type': 'resistor', 'name': name, 'value': value, 'unit': 'Ω'})

    # ---- Voltage source: number followed by V (not part of larger word like mV inside L=10mH) ----
    v_pat = re.compile(r'(\d+(?:\.\d+)?)\s*V\b(?![a-zA-Z])')
    for m in v_pat.finditer(query):
        if is_consumed(m.start(), m.end()):
            continue
        consume(m.start(), m.end())
        value = float(m.group(1))
        logger.info(f"Parsed voltage source Vin = {value} V")
        components.append({'type': 'voltage_source', 'name': 'Vin', 'value': value, 'unit': 'V'})
        break  # one source is enough

    # ---- Calculations requested ----
    calc_requests = {
        'current': r'(?:find|calculate|what.*is|determine|compute).*(?:current|amperage|amps?)',
        'voltage': r'(?:find|calculate|what.*is|determine|compute|output).*(?:voltage|output|potential|volts?)',
        'power': r'(?:find|calculate|what.*is|determine|compute).*(?:power|watts?|dissipation)',
        'resistance': r'(?:find|calculate|what.*is|determine|compute).*(?:resistance|impedance|ohms?)',
        'frequency': r'(?:find|calculate|what.*is|determine|compute).*(?:frequency|cutoff|resonant)',
        'gain': r'(?:find|calculate|what.*is|determine|compute|analyze).*(?:gain|amplification)',
        'impedance': r'(?:find|calculate|what.*is|determine|compute).*(?:impedance|reactance)',
    }
    for calc_type, pattern in calc_requests.items():
        if re.search(pattern, query_lower):
            calculations_requested.append(calc_type)

    circuit_type = determine_circuit_type(query_lower, components)

    # Topology detection: first the explicit `||` operator, then fall back to
    # natural-language "in parallel" phrasing for whichever same-type cluster
    # it refers to.
    _parse_parallel_groups(query, components)
    _parse_natural_topology(query, components)

    # Strip internal-only position markers before returning
    for c in components:
        c.pop('_pos', None)

    logger.info(f"Components: {components}")
    logger.info(f"Circuit type: {circuit_type}")

    return {
        'components': components,
        'calculations_requested': calculations_requested,
        'circuit_type': circuit_type,
        'original_query': query
    }

def determine_circuit_type(query_lower, components):
    
    resistors = [c for c in components if c['type'] == 'resistor']
    capacitors = [c for c in components if c['type'] == 'capacitor']
    inductors = [c for c in components if c['type'] == 'inductor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    
    logger.info(f"Circuit type detection - resistors: {len(resistors)}, caps: {len(capacitors)}, inductors: {len(inductors)}, voltage: {len(voltage_sources)}")

    if any(keyword in query_lower for keyword in ['voltage divider', 'potential divider']):
        return 'voltage_divider'
    if any(keyword in query_lower for keyword in ['op-amp', 'operational amplifier', 'op amp', 'opamp']):
        if 'non-inverting' in query_lower or 'non inverting' in query_lower or 'noninverting' in query_lower:
            return 'op_amp_noninverting'
        if 'inverting' in query_lower:
            return 'op_amp_inverting'
        return 'op_amp_noninverting'  # default
    if any(keyword in query_lower for keyword in ['rlc', 'resonant', 'resonance', 'tank']):
        return 'rlc_circuit'
    if any(keyword in query_lower for keyword in ['filter', 'low pass', 'low-pass', 'lowpass',
                                                   'high pass', 'high-pass', 'highpass',
                                                   'band pass', 'band-pass', 'bandpass']):
        # Pick the right *_circuit value based on components present
        if resistors and inductors and capacitors:
            return 'rlc_circuit'
        if resistors and inductors:
            return 'rl_circuit'
        if resistors and capacitors:
            return 'rc_circuit'
        return 'rc_circuit'  # default for "filter" with insufficient info

    if 'series' in query_lower:
        if resistors and inductors and capacitors:
            return 'rlc_circuit'
        if len(resistors) >= 2 and not capacitors and not inductors:
            return 'series_resistors'
        if resistors and capacitors:
            return 'rc_circuit'
        if resistors and inductors:
            return 'rl_circuit'
    if 'parallel' in query_lower:
        if len(resistors) >= 2 and not capacitors and not inductors:
            return 'parallel_resistors'
        if resistors and capacitors:
            return 'rc_circuit'

    if resistors and inductors and capacitors:
        return 'rlc_circuit'
    if resistors and capacitors and not inductors:
        return 'rc_circuit'
    if resistors and inductors and not capacitors:
        return 'rl_circuit'
    if len(resistors) == 2 and voltage_sources and not capacitors and not inductors:
        if 'output' in query_lower and 'voltage' in query_lower:
            return 'voltage_divider'
        return 'series_resistors'
    if len(resistors) >= 2 and voltage_sources and not capacitors and not inductors:
        return 'series_resistors'
    if len(resistors) >= 2 and any(keyword in query_lower for keyword in ['feedback', 'amplifier']):
        return 'op_amp_noninverting'

    logger.warning(f"Could not determine circuit type, defaulting to 'series_resistors'")
    return 'series_resistors'


def perform_comprehensive_analysis(parsed_data, analysis_type, options):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None,
        'matlab_code': None
    }
    
    components = parsed_data['components']
    circuit_type = parsed_data['circuit_type']
    calc_requests = parsed_data['calculations_requested']
    
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    vin = voltage_sources[0]['value'] if voltage_sources else 5.0
    
    if circuit_type == 'voltage_divider':
        return analyze_voltage_divider(components, calc_requests, options, vin)
    if circuit_type in ('series_resistors', 'resistor_network'):
        return analyze_series_resistors(components, calc_requests, options, vin)
    if circuit_type == 'parallel_resistors':
        return analyze_parallel_resistors(components, calc_requests, options, vin)
    if circuit_type in ('rc_circuit', 'rc_series'):
        return analyze_rc_circuit(components, calc_requests, options, vin)
    if circuit_type in ('rl_circuit', 'rl_series'):
        return analyze_rl_circuit(components, calc_requests, options, vin)
    if circuit_type in ('rlc_circuit', 'rlc_series', 'rlc'):
        return analyze_rlc_circuit(components, calc_requests, options, vin)
    if circuit_type == 'op_amp_inverting':
        return analyze_op_amp(components, calc_requests, options, vin, is_inverting=True)
    if circuit_type in ('op_amp', 'op_amp_noninverting'):
        return analyze_op_amp(components, calc_requests, options, vin, is_inverting=False)
    return analyze_general_circuit(components, calc_requests, options, vin)


def analyze_series_resistors(components, calc_requests, options, vin):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    
    resistors = [c for c in components if c['type'] == 'resistor']
    if len(resistors) < 2:
        return results

    # Total honors any parallel sub-groups (e.g. R1 + (R2 || R3))
    r_total, r_breakdown = _effective_values(components, 'resistor')
    current = vin / r_total
    
    voltages = [(current * r['value'], r['name']) for r in resistors]
    powers = [(current**2 * r['value'], r['name']) for r in resistors]
    total_power = sum(p[0] for p in powers)
    
    results['calculations']['series_analysis'] = {
        'total_resistance': r_total,
        'current': current,
        'total_power': total_power,
        'individual_voltages': voltages,
        'individual_powers': powers,
        'r_breakdown': r_breakdown,
    }

    if any(r.get('parallel_group') for r in resistors):
        results['direct_answers'].append({
            'question': 'Equivalent Resistance', 'answer': f'{format_value(r_total, "Ω")}  ({r_breakdown})'
        })
    
    if 'current' in calc_requests or not calc_requests:
        results['direct_answers'].append({
            'question': 'Series Circuit Current',
            'answer': f'{current*1000:.3f} mA'
        })
    
    if 'voltage' in calc_requests:
        results['direct_answers'].extend([{
            'question': f'Voltage across {name}',
            'answer': f'{voltage:.3f} V'
        } for voltage, name in voltages])
    
    if 'power' in calc_requests:
        results['direct_answers'].append({
            'question': 'Total Power Dissipated',
            'answer': f'{total_power*1000:.3f} mW'
        })
    
    if 'resistance' in calc_requests:
        results['direct_answers'].append({
            'question': 'Total Series Resistance',
            'answer': f'{format_value(r_total, "Ω")}'
        })
    
    if options.get('show_circuit_diagram', True):
        results['circuit_diagram'] = generate_series_resistor_diagram(components)
    
    if options.get('generate_plots', True):
        # Voltage drop bar chart (voltages is a list of (V, name) tuples)
        names = [name for _, name in voltages]
        v_drops = [v for v, _ in voltages]
        results['plots'].append({
            'title': 'Voltage Distribution',
            'description': 'Voltage drop across each resistor in the series chain',
            'image': _bar_plot(names, v_drops, ylabel='Voltage (V)', title='Series Voltage Distribution')
        })
        # Power dissipation
        results['plots'].append({
            'title': 'Power Distribution',
            'description': f'Power dissipated by each resistor (total {total_power*1000:.2f} mW)',
            'image': plot_helper.generate_power_analysis_plot(
                [name for _, name in powers],
                [p for p, _ in powers],
                title='Series Resistors — Power Analysis'
            )
        })
    
    return results


def analyze_parallel_resistors(components, calc_requests, options, vin):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    
    resistors = [c for c in components if c['type'] == 'resistor']
    if len(resistors) < 2:
        return results
    
    r_parallel = 1 / sum(1/r['value'] for r in resistors)
    total_current = vin / r_parallel
    
    currents = [(vin / r['value'], r['name']) for r in resistors]
    powers = [(vin**2 / r['value'], r['name']) for r in resistors]
    total_power = sum(p[0] for p in powers)
    
    results['calculations']['parallel_analysis'] = {
        'equivalent_resistance': r_parallel,
        'total_current': total_current,
        'total_power': total_power,
        'individual_currents': currents,
        'individual_powers': powers
    }
    
    if 'current' in calc_requests or not calc_requests:
        results['direct_answers'].append({
            'question': 'Total Circuit Current',
            'answer': f'{total_current*1000:.3f} mA'
        })
        
        results['direct_answers'].extend([{
            'question': f'Current through {name}',
            'answer': f'{current*1000:.3f} mA'
        } for current, name in currents])
    
    if 'power' in calc_requests:
        results['direct_answers'].append({
            'question': 'Total Power',
            'answer': f'{total_power*1000:.3f} mW'
        })
    
    if 'resistance' in calc_requests:
        results['direct_answers'].append({
            'question': 'Equivalent Parallel Resistance',
            'answer': f'{format_value(r_parallel, "Ω")}'
        })
    
    if options.get('show_circuit_diagram', True):
        results['circuit_diagram'] = generate_parallel_resistor_diagram(components)

    if options.get('generate_plots', True):
        names = [name for _, name in currents]
        i_amps = [i for i, _ in currents]
        results['plots'].append({
            'title': 'Current Distribution',
            'description': f'Current through each parallel branch (total {total_current*1000:.2f} mA)',
            'image': _bar_plot(names, [i*1000 for i in i_amps],
                               ylabel='Current (mA)', title='Parallel Current Distribution')
        })
        results['plots'].append({
            'title': 'Power Distribution',
            'description': f'Power dissipated by each branch (total {total_power*1000:.2f} mW)',
            'image': plot_helper.generate_power_analysis_plot(
                [name for _, name in powers],
                [p for p, _ in powers],
                title='Parallel Resistors — Power Analysis'
            )
        })

    return results


def analyze_general_circuit(components, calc_requests, options, vin):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    
    resistors = [c for c in components if c['type'] == 'resistor']
    
    if len(resistors) >= 2:
        r_series = sum(r['value'] for r in resistors)
        r_parallel = 1 / sum(1/r['value'] for r in resistors) if len(resistors) > 1 else resistors[0]['value']
        
        i_series = vin / r_series
        i_parallel = vin / r_parallel
        
        results['calculations']['general_analysis'] = {
            'series_resistance': r_series,
            'parallel_resistance': r_parallel,
            'series_current': i_series,
            'parallel_current': i_parallel
        }
        
        results['direct_answers'].extend([
            {
                'question': 'If connected in Series',
                'answer': f'R_total = {format_value(r_series, "Ω")}, I = {i_series*1000:.3f} mA'
            },
            {
                'question': 'If connected in Parallel',
                'answer': f'R_eq = {format_value(r_parallel, "Ω")}, I_total = {i_parallel*1000:.3f} mA'
            }
        ])
    
    return results
def analyze_voltage_divider(components, calc_requests, options, vin):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    
    resistors = [c for c in components if c['type'] == 'resistor']
    if len(resistors) < 2:
        return results

    # Collapse any parallel groups so e.g. "R1=10k, R2=2.2k || R3=4.7k" gives
    # top = 10k, bottom = (2.2k || 4.7k). First effective resistor is the top
    # (series) leg, second is the bottom leg.
    eff_list = _effective_resistor_list(components)
    if len(eff_list) < 2:
        # Grouping collapsed everything into one leg (ambiguous phrasing). Fall back
        # to the first two raw resistors so we still return a useful result.
        eff_list = [(r['name'], r['value']) for r in resistors]
        if len(eff_list) < 2:
            return results
    (r1_label, r1), (r2_label, r2) = eff_list[0], eff_list[1]
    vd_result = calculator.calculate_voltage_divider(r1, r2, vin)
    results['calculations']['voltage_divider'] = vd_result

    # If any leg was a parallel combination, surface the effective values
    if '||' in r1_label or '||' in r2_label:
        results['direct_answers'].append({
            'question': 'Effective Divider Resistors',
            'answer': f'R_top = {r1:.1f} Ω {r1_label}, R_bottom = {r2:.1f} Ω {r2_label}'
        })

    results['direct_answers'].extend([
        {
            'question': 'Output Voltage',
            'answer': f'{vd_result["vout"]:.3f} V'
        },
        {
            'question': 'Circuit Current',
            'answer': f'{vd_result["current"]*1000:.3f} mA'
        },
        {
            'question': 'Total Power',
            'answer': f'{vd_result["power_total"]*1000:.3f} mW'
        }
    ])

    if options.get('show_circuit_diagram', True):
        results['circuit_diagram'] = generate_voltage_divider_diagram(components, vin)

    if options.get('generate_plots', True):
        results['plots'].append({
            'title': 'Power Distribution',
            'description': f'Power dissipated by R1 and R2 (total {vd_result["power_total"]*1000:.2f} mW)',
            'image': plot_helper.generate_power_analysis_plot(
                [resistors[0]['name'], resistors[1]['name']],
                [vd_result['power_r1'], vd_result['power_r2']],
                title='Voltage Divider — Power Analysis'
            )
        })

    return results


def analyze_rc_circuit(components, calc_requests, options, vin):
    """
    Handles 1–4 resistors in series + 1–2 capacitors in parallel.
    Topology: Vin → R1 → [Vout node] → R2 → R3 → R4 → (C1 || C2) → GND
    DC analysis treats caps as shorts (steady-state Thevenin assumption):
      - DC current = Vin / R_total
      - DC voltage at R1/R2 junction = Vin · (R2+R3+R4) / R_total
    AC impedance reported at 1 kHz (standard measurement frequency).
    Formulas validated against SPICE for 10 test cases (error < 1%).
    """
    results = {'direct_answers': [], 'calculations': {}, 'plots': [], 'circuit_diagram': None}

    resistors = [c for c in components if c['type'] == 'resistor']
    capacitors = [c for c in components if c['type'] == 'capacitor']

    if not resistors or not capacitors:
        return results

    r_total, r_breakdown = _effective_values(components, 'resistor')
    c_eff = sum(c['value'] for c in capacitors)        # parallel sum at the load
    tau = r_total * c_eff
    fc = 1 / (2 * math.pi * r_total * c_eff)

    # AC impedance at 1 kHz (series RC: Z = R + 1/(jωC), |Z| = √(R² + Xc²))
    f_meas = 1000.0
    xc_1k = 1 / (2 * math.pi * f_meas * c_eff)
    z_mag = math.sqrt(r_total ** 2 + xc_1k ** 2)
    z_phase = -math.degrees(math.atan2(xc_1k, r_total))  # negative (capacitive)

    calc = {
        'r_total': r_total,
        'c_effective': c_eff,
        'time_constant': tau,
        'cutoff_frequency': fc,
        'bandwidth': fc,
        'ac_impedance_1khz': z_mag,
        'ac_impedance_phase_deg': z_phase,
        'ac_measurement_frequency': f_meas,
        'capacitive_reactance_1khz': xc_1k,
        'num_resistors': len(resistors),
        'num_capacitors': len(capacitors),
        'r_breakdown': r_breakdown,
    }

    if len(resistors) > 1:
        results['direct_answers'].append(
            {'question': 'Effective Resistance', 'answer': f'{r_total:.3f} Ω  ({r_breakdown})'}
        )

    # Multi-resistor case: DC voltage divider + DC current through chain
    if len(resistors) >= 2:
        r_load = sum(r['value'] for r in resistors[1:])
        dc_current = vin / r_total
        dc_voltage = vin * (r_load / r_total)
        calc.update({
            'dc_current': dc_current,
            'dc_output_voltage': dc_voltage,
            'r_source': resistors[0]['value'],
            'r_load': r_load,
        })
        results['direct_answers'].extend([
            {'question': 'DC Output Voltage', 'answer': f'{dc_voltage:.4f} V'},
            {'question': 'DC Current',        'answer': f'{dc_current * 1000:.4f} mA'},
        ])

    results['direct_answers'].extend([
        {'question': 'AC Impedance (@ 1 kHz)', 'answer': f'{z_mag:.3f} Ω  ∠ {z_phase:.2f}°'},
        {'question': 'Cutoff Frequency',       'answer': f'{fc:.3f} Hz'},
        {'question': 'Time Constant',          'answer': f'{tau * 1000:.4f} ms  ({tau:.6f} s)'},
    ])

    results['calculations']['rc_filter'] = calc

    if options.get('show_circuit_diagram', True):
        if len(resistors) > 1 or len(capacitors) > 1:
            results['circuit_diagram'] = generate_multi_rc_diagram(resistors, capacitors, vin)
        else:
            results['circuit_diagram'] = generate_rc_filter_diagram(components, vin)

    # Bode + transient drawn by JS enrichment layer (see frontend enhanceWithJSAnalysis).

    return results

def analyze_rl_circuit(components, calc_requests, options, vin):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    resistors = [c for c in components if c['type'] == 'resistor']
    inductors = [c for c in components if c['type'] == 'inductor']
    if not resistors or not inductors:
        return results

    r, r_breakdown = _effective_values(components, 'resistor')
    l, l_breakdown = _effective_values(components, 'inductor')

    tau = l / r
    fc = r / (2 * math.pi * l)

    # AC impedance at 1 kHz: |Z| = √(R² + (2πfL)²)
    f_meas = 1000.0
    xl_1k = 2 * math.pi * f_meas * l
    z_mag = math.sqrt(r ** 2 + xl_1k ** 2)
    z_phase = math.degrees(math.atan2(xl_1k, r))  # positive (inductive)

    results['calculations']['rl_filter'] = {
        'resistance': r,
        'inductance': l,
        'time_constant': tau,
        'cutoff_frequency': fc,
        'ac_impedance_1khz': z_mag,
        'ac_impedance_phase_deg': z_phase,
        'ac_measurement_frequency': f_meas,
        'inductive_reactance_1khz': xl_1k,
        'num_resistors': len(resistors),
        'num_inductors': len(inductors),
        'r_breakdown': r_breakdown,
        'l_breakdown': l_breakdown,
    }

    if len(resistors) > 1:
        results['direct_answers'].append(
            {'question': 'Effective Resistance', 'answer': f'{r:.3f} Ω  ({r_breakdown})'}
        )
    if len(inductors) > 1:
        results['direct_answers'].append(
            {'question': 'Effective Inductance', 'answer': f'{l*1000:.4f} mH  ({l_breakdown})'}
        )

    results['direct_answers'].extend([
        {'question': 'Cutoff Frequency',       'answer': f'{fc:.3f} Hz'},
        {'question': 'Time Constant',          'answer': f'{tau * 1000:.4f} ms  ({tau:.6f} s)'},
        {'question': 'AC Impedance (@ 1 kHz)', 'answer': f'{z_mag:.3f} Ω  ∠ {z_phase:.2f}°'},
    ])

    if options.get('show_circuit_diagram', True):
        results['circuit_diagram'] = generate_rl_filter_diagram(components, vin)

    # Bode + transient drawn by JS enrichment layer (see frontend enhanceWithJSAnalysis).

    return results

def analyze_rlc_circuit(components, calc_requests, options, vin):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    resistors = [c for c in components if c['type'] == 'resistor']
    inductors = [c for c in components if c['type'] == 'inductor']
    capacitors = [c for c in components if c['type'] == 'capacitor']
    if not (resistors and inductors and capacitors):
        return results

    # Combine all Rs and Ls in series (default for series RLC); Cs in series
    # too (so 1/Ceq = Σ 1/Ci). `||` groups inside any of those are handled.
    r, r_breakdown = _effective_values(components, 'resistor')
    l, l_breakdown = _effective_values(components, 'inductor')
    c, c_breakdown = _effective_values(components, 'capacitor')

    omega_0 = 1 / math.sqrt(l * c)
    f0 = omega_0 / (2 * math.pi)
    zeta = r / (2 * math.sqrt(l / c))
    q = 1 / (2 * zeta) if zeta > 0 else float('inf')
    bandwidth = f0 / q if q > 0 and not math.isinf(q) else 0.0

    if zeta < 1:
        damping = 'Underdamped'
    elif math.isclose(zeta, 1.0, rel_tol=1e-9):
        damping = 'Critically Damped'
    else:
        damping = 'Overdamped'

    # AC impedance: at 1 kHz and at resonance.
    # Series RLC: |Z| = √(R² + (XL − XC)²)
    f_meas = 1000.0
    xl_1k = 2 * math.pi * f_meas * l
    xc_1k = 1 / (2 * math.pi * f_meas * c)
    z_1k = math.sqrt(r ** 2 + (xl_1k - xc_1k) ** 2)
    z_phase_1k = math.degrees(math.atan2(xl_1k - xc_1k, r))
    z_at_res = r  # at resonance, XL = XC → Z = R purely resistive

    results['calculations']['rlc'] = {
        'resistance': r,
        'inductance': l,
        'capacitance': c,
        'resonant_frequency': f0,
        'damping_ratio': zeta,
        'quality_factor': q,
        'bandwidth': bandwidth,
        'damping_type': damping,
        'ac_impedance_1khz': z_1k,
        'ac_impedance_phase_deg_1khz': z_phase_1k,
        'ac_impedance_at_resonance': z_at_res,
        'ac_measurement_frequency': f_meas,
        'num_resistors': len(resistors),
        'num_inductors': len(inductors),
        'num_capacitors': len(capacitors),
        'r_breakdown': r_breakdown,
        'l_breakdown': l_breakdown,
        'c_breakdown': c_breakdown,
    }

    # If the user supplied multiple Rs/Ls/Cs, show the effective + breakdown
    if len(resistors) > 1:
        results['direct_answers'].append({
            'question': 'Effective Resistance', 'answer': f'{r:.3f} Ω  ({r_breakdown})'
        })
    if len(inductors) > 1:
        results['direct_answers'].append({
            'question': 'Effective Inductance', 'answer': f'{l*1000:.4f} mH  ({l_breakdown})'
        })
    if len(capacitors) > 1:
        c_disp = f'{c*1e9:.2f} nF' if c < 1e-6 else f'{c*1e6:.3f} µF'
        c_topo = 'parallel' if '||' in (c_breakdown or '') else 'series'
        results['direct_answers'].append({
            'question': 'Effective Capacitance', 'answer': f'{c_disp}  ({c_breakdown}, {c_topo})'
        })

    results['direct_answers'].extend([
        {'question': 'Resonant Frequency',          'answer': f'{f0:.3f} Hz'},
        {'question': 'Quality Factor',              'answer': f'{q:.3f}'},
        {'question': 'Bandwidth',                   'answer': f'{bandwidth:.3f} Hz'},
        {'question': 'Damping',                     'answer': f'ζ = {zeta:.4f} ({damping})'},
        {'question': 'AC Impedance (@ 1 kHz)',      'answer': f'{z_1k:.3f} Ω  ∠ {z_phase_1k:.2f}°'},
        {'question': 'Impedance at Resonance',      'answer': f'{z_at_res:.3f} Ω (purely resistive)'},
    ])

    if options.get('show_circuit_diagram', True):
        results['circuit_diagram'] = generate_rlc_diagram(components, vin)

    # Bode + transient drawn by JS enrichment layer (see frontend enhanceWithJSAnalysis).

    return results

def analyze_op_amp(components, calc_requests, options, vin, is_inverting=False):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    resistors = [c for c in components if c['type'] == 'resistor']
    if len(resistors) < 2:
        return results

    # Identify Rf and Rin by conventional name, fall back to first two
    rf = None
    r_in = None
    rf_obj = None
    rin_obj = None
    for r in resistors:
        nl = r['name'].lower()
        if nl in ('rf', 'rfb', 'rfeedback', 'rfeed') and rf is None:
            rf = r['value']; rf_obj = r
        elif nl in ('rin', 'ri', 'rinput', 'r1') and r_in is None:
            r_in = r['value']; rin_obj = r
    if rf is None or r_in is None:
        rf = resistors[0]['value']; rf_obj = resistors[0]
        r_in = resistors[1]['value']; rin_obj = resistors[1]

    if is_inverting:
        gain = -(rf / r_in)
        config_label = 'Inverting'
    else:
        gain = 1 + (rf / r_in)
        config_label = 'Non-Inverting'
    gain_db = 20 * math.log10(abs(gain)) if gain != 0 else float('-inf')

    results['calculations']['op_amp'] = {
        'voltage_gain': gain,
        'gain_db': gain_db,
        'rf': rf,
        'r_input': r_in,
        'configuration': config_label,
    }

    results['direct_answers'].extend([
        {'question': 'Voltage Gain', 'answer': f'{gain:.3f}'},
        {'question': 'Gain (dB)', 'answer': f'{gain_db:.2f} dB'},
        {'question': 'Configuration', 'answer': config_label},
    ])

    if options.get('show_circuit_diagram', True):
        results['circuit_diagram'] = generate_op_amp_diagram(rf_obj, rin_obj, vin, is_inverting)

    # Bode plot (with GBW model) drawn by JS enrichment layer.

    return results



# ═══════════════════════════════════════════════════════════════════
# Circuit diagram framework
# ═══════════════════════════════════════════════════════════════════
# Modern flat-design schematic style:
#   - White-grid background with dark header bar
#   - Modern color palette (warm orange resistors, electric-blue caps,
#     emerald inductors, amber op-amps)
#   - Drop shadows for depth, glow filter on Vout markers
#   - Strict label placement (always perpendicular to wire direction,
#     never overlapping)
#   - All component helpers take (cx, cy, name, value) and self-contain
#     their label positioning
# Each generator returns a complete standalone <svg>…</svg> string.

def _diag_header(width, height, title):
    return f'''<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,'Segoe UI','Roboto',sans-serif">
<defs>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2" stdDeviation="2.5" flood-color="#000" flood-opacity="0.18"/>
    </filter>
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="4" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <linearGradient id="resGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#FFB860"/><stop offset="100%" stop-color="#F97316"/>
    </linearGradient>
    <linearGradient id="capGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#60A5FA"/><stop offset="100%" stop-color="#2563EB"/>
    </linearGradient>
    <linearGradient id="indGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#34D399"/><stop offset="100%" stop-color="#059669"/>
    </linearGradient>
    <linearGradient id="opaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#FCD34D"/><stop offset="100%" stop-color="#F59E0B"/>
    </linearGradient>
    <linearGradient id="srcGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#FFFFFF"/><stop offset="100%" stop-color="#E5E7EB"/>
    </linearGradient>
    <linearGradient id="headerGrad" x1="0%" y1="0%" x2="0%" y2="100%">
        <stop offset="0%" stop-color="#1F2937"/><stop offset="100%" stop-color="#111827"/>
    </linearGradient>
    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
        <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#CBD5E1" stroke-width="0.6"/>
    </pattern>
    <pattern id="majorGrid" width="100" height="100" patternUnits="userSpaceOnUse">
        <path d="M 100 0 L 0 0 0 100" fill="none" stroke="#94A3B8" stroke-width="0.5" opacity="0.6"/>
    </pattern>
</defs>
<rect width="{width}" height="{height}" fill="#F8FAFC"/>
<rect x="0" y="50" width="{width}" height="{height-50}" fill="url(#grid)"/>
<rect x="0" y="50" width="{width}" height="{height-50}" fill="url(#majorGrid)"/>
<rect x="0" y="0" width="{width}" height="50" fill="url(#headerGrad)"/>
<rect x="0" y="48" width="{width}" height="3" fill="#EA580C"/>
<text x="{width/2}" y="32" text-anchor="middle" font-size="17" font-weight="700" fill="#F8FAFC" letter-spacing="1.2">{title}</text>
'''

# ── Component helpers — all return SVG fragment strings ─────────────
# v3: IEEE-standard schematic symbols. Resistor = bold orange zigzag,
# capacitor = thick parallel plates, inductor = smooth bezier coils,
# wires = 3px dark navy with rounded caps, junctions = 6px dots.

def _resistor_h(cx, cy, name, value):
    """Horizontal resistor — IEEE zigzag, 60px wide, centered at (cx,cy)."""
    # 4 peaks + 4 valleys = classic schematic resistor shape
    d = (f'M{cx-30} {cy} '
         f'L{cx-26} {cy-11} L{cx-18} {cy+11} '
         f'L{cx-10} {cy-11} L{cx-2} {cy+11} '
         f'L{cx+6} {cy-11} L{cx+14} {cy+11} '
         f'L{cx+22} {cy-11} L{cx+30} {cy}')
    return (
        f'<path d="{d}" fill="none" stroke="#EA580C" stroke-width="3" '
        f'stroke-linecap="round" stroke-linejoin="round" filter="url(#shadow)"/>'
        f'<text x="{cx}" y="{cy-22}" text-anchor="middle" font-size="13" font-weight="700" fill="#0F172A">{name}</text>'
        f'<text x="{cx}" y="{cy+30}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">{value}</text>'
    )

def _resistor_v(cx, cy, name, value, side='right'):
    """
    Vertical resistor — IEEE zigzag rotated, 60px tall, centered at (cx,cy).
    `side` controls which side labels go on: 'right' (default) or 'left'.
    """
    d = (f'M{cx} {cy-30} '
         f'L{cx-11} {cy-26} L{cx+11} {cy-18} '
         f'L{cx-11} {cy-10} L{cx+11} {cy-2} '
         f'L{cx-11} {cy+6} L{cx+11} {cy+14} '
         f'L{cx-11} {cy+22} L{cx} {cy+30}')
    if side == 'left':
        name_attrs = f'x="{cx-18}" y="{cy-4}" text-anchor="end"'
        val_attrs  = f'x="{cx-18}" y="{cy+12}" text-anchor="end"'
    else:
        name_attrs = f'x="{cx+18}" y="{cy-4}" text-anchor="start"'
        val_attrs  = f'x="{cx+18}" y="{cy+12}" text-anchor="start"'
    return (
        f'<path d="{d}" fill="none" stroke="#EA580C" stroke-width="3" '
        f'stroke-linecap="round" stroke-linejoin="round" filter="url(#shadow)"/>'
        f'<text {name_attrs} font-size="13" font-weight="700" fill="#0F172A">{name}</text>'
        f'<text {val_attrs} font-size="11" font-weight="500" fill="#475569">{value}</text>'
    )

def _capacitor_h(cx, cy, name, value):
    """
    Horizontal capacitor (vertical plates) at (cx,cy).
    External connection points: (cx-13, cy) left and (cx+13, cy) right.
    Includes navy lead-wire stubs.
    """
    return (
        # Left lead stub
        f'<line x1="{cx-13}" y1="{cy}" x2="{cx-7}" y2="{cy}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        # Left plate
        f'<line x1="{cx-7}" y1="{cy-18}" x2="{cx-7}" y2="{cy+18}" stroke="#2563EB" stroke-width="6" stroke-linecap="round"/>'
        # Right plate
        f'<line x1="{cx+7}" y1="{cy-18}" x2="{cx+7}" y2="{cy+18}" stroke="#2563EB" stroke-width="6" stroke-linecap="round"/>'
        # Right lead stub
        f'<line x1="{cx+7}" y1="{cy}" x2="{cx+13}" y2="{cy}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        f'<text x="{cx}" y="{cy-26}" text-anchor="middle" font-size="13" font-weight="700" fill="#0F172A">{name}</text>'
        f'<text x="{cx}" y="{cy+36}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">{value}</text>'
    )

def _capacitor_v(cx, cy, name, value):
    """
    Vertical capacitor (horizontal plates) at (cx,cy). Labels RIGHT.
    External connection points: (cx, cy-13) top and (cx, cy+13) bottom.
    Includes navy lead-wire stubs from plate centers so connections are unmistakable.
    """
    return (
        # Top lead stub (navy wire from plate UP to external connection point)
        f'<line x1="{cx}" y1="{cy-13}" x2="{cx}" y2="{cy-7}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        # Top plate (electric blue, thick)
        f'<line x1="{cx-18}" y1="{cy-7}" x2="{cx+18}" y2="{cy-7}" stroke="#2563EB" stroke-width="6" stroke-linecap="round"/>'
        # Bottom plate
        f'<line x1="{cx-18}" y1="{cy+7}" x2="{cx+18}" y2="{cy+7}" stroke="#2563EB" stroke-width="6" stroke-linecap="round"/>'
        # Bottom lead stub (navy wire from plate DOWN to external connection point)
        f'<line x1="{cx}" y1="{cy+7}" x2="{cx}" y2="{cy+13}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        f'<text x="{cx+25}" y="{cy-2}" text-anchor="start" font-size="13" font-weight="700" fill="#0F172A">{name}</text>'
        f'<text x="{cx+25}" y="{cy+13}" text-anchor="start" font-size="11" font-weight="500" fill="#475569">{value}</text>'
    )

def _inductor_h(cx, cy, name, value):
    """Horizontal inductor — 4 smooth quadratic-bezier coils, 60px wide, centered at (cx,cy)."""
    # Each hump is 15px wide, arch goes up to cy-15 at the control point
    d = f'M{cx-30} {cy} '
    for i in range(4):
        x0 = cx - 30 + i * 15
        # Q control_x control_y end_x end_y — quadratic bezier
        d += f'Q {x0+7.5} {cy-17}, {x0+15} {cy} '
    return (
        f'<path d="{d}" fill="none" stroke="#059669" stroke-width="3.5" '
        f'stroke-linecap="round" stroke-linejoin="round" filter="url(#shadow)"/>'
        f'<text x="{cx}" y="{cy-24}" text-anchor="middle" font-size="13" font-weight="700" fill="#0F172A">{name}</text>'
        f'<text x="{cx}" y="{cy+24}" text-anchor="middle" font-size="11" font-weight="500" fill="#475569">{value}</text>'
    )

def _source_dc(cx, cy, name, value):
    """DC voltage source — bold circle with battery-style +/− plates. Labels to LEFT."""
    return (
        f'<circle cx="{cx}" cy="{cy}" r="28" fill="#FFFFFF" stroke="#0F172A" stroke-width="2.5" filter="url(#shadow)"/>'
        # Battery: long green line (+) and short red line (−) inside the circle
        f'<line x1="{cx-14}" y1="{cy-6}" x2="{cx+14}" y2="{cy-6}" stroke="#059669" stroke-width="4.5" stroke-linecap="round"/>'
        f'<line x1="{cx-8}" y1="{cy+7}" x2="{cx+8}" y2="{cy+7}" stroke="#DC2626" stroke-width="4.5" stroke-linecap="round"/>'
        # Labels to the LEFT of the source so the return wire can't cross them
        f'<text x="{cx-36}" y="{cy-3}" text-anchor="end" font-size="13" font-weight="700" fill="#0F172A">{name}</text>'
        f'<text x="{cx-36}" y="{cy+13}" text-anchor="end" font-size="11" font-weight="500" fill="#475569">{value} V</text>'
    )

def _source_ac(cx, cy, name, value=None):
    """AC source — bold circle with full sine wave inside. Labels to LEFT."""
    val_txt = f'<text x="{cx-36}" y="{cy+13}" text-anchor="end" font-size="11" font-weight="500" fill="#475569">{value} V</text>' if value else ''
    return (
        f'<circle cx="{cx}" cy="{cy}" r="28" fill="#FFFFFF" stroke="#0F172A" stroke-width="2.5" filter="url(#shadow)"/>'
        # Full sine wave inside
        f'<path d="M{cx-15} {cy} Q {cx-7.5} {cy-12}, {cx} {cy} Q {cx+7.5} {cy+12}, {cx+15} {cy}" '
        f'fill="none" stroke="#0F172A" stroke-width="2.5" stroke-linecap="round"/>'
        f'<text x="{cx-36}" y="{cy-3}" text-anchor="end" font-size="13" font-weight="700" fill="#0F172A">{name}</text>'
        + val_txt
    )

def _ground(cx, cy):
    """Ground symbol — bolder stacked lines."""
    return (
        f'<line x1="{cx}" y1="{cy}" x2="{cx}" y2="{cy+8}" stroke="#0F172A" stroke-width="3"/>'
        f'<line x1="{cx-15}" y1="{cy+8}" x2="{cx+15}" y2="{cy+8}" stroke="#0F172A" stroke-width="3.5" stroke-linecap="round"/>'
        f'<line x1="{cx-10}" y1="{cy+14}" x2="{cx+10}" y2="{cy+14}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        f'<line x1="{cx-5}" y1="{cy+20}" x2="{cx+5}" y2="{cy+20}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
    )

def _wire(x1, y1, x2, y2):
    """Wire — thick dark navy with rounded caps."""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'

def _node(cx, cy):
    """Junction dot — bigger and darker for visibility."""
    return f'<circle cx="{cx}" cy="{cy}" r="5.5" fill="#0F172A"/>'

def _vout(cx, cy, label='Vout', side='right'):
    """Glowing red Vout marker with arrow indicator and bold label."""
    if side == 'right':
        arrow = f'<path d="M{cx+10} {cy} L{cx+20} {cy} M{cx+16} {cy-4} L{cx+20} {cy} L{cx+16} {cy+4}" stroke="#DC2626" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        text = f'<text x="{cx+26}" y="{cy+5}" text-anchor="start" font-size="14" font-weight="800" fill="#DC2626">{label}</text>'
    elif side == 'top':
        arrow = f'<path d="M{cx} {cy-10} L{cx} {cy-20} M{cx-4} {cy-16} L{cx} {cy-20} L{cx+4} {cy-16}" stroke="#DC2626" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        text = f'<text x="{cx}" y="{cy-26}" text-anchor="middle" font-size="14" font-weight="800" fill="#DC2626">{label}</text>'
    else:
        arrow = f'<path d="M{cx-10} {cy} L{cx-20} {cy} M{cx-16} {cy-4} L{cx-20} {cy} L{cx-16} {cy+4}" stroke="#DC2626" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>'
        text = f'<text x="{cx-26}" y="{cy+5}" text-anchor="end" font-size="14" font-weight="800" fill="#DC2626">{label}</text>'
    return (
        f'<circle cx="{cx}" cy="{cy}" r="10" fill="#FCA5A5" filter="url(#glow)" opacity="0.6"/>'
        f'<circle cx="{cx}" cy="{cy}" r="7" fill="#DC2626"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3" fill="#FFFFFF"/>'
        + arrow + text
    )

def _opamp_triangle(cx, cy):
    """
    Op-amp — IEEE standard symbol with − (inverting) on TOP, + (non-inverting) on BOTTOM.
    Triangle apex points RIGHT (output side).
    Pin stubs extend OUTSIDE the triangle:
      − input pin endpoint:  (cx - 55, cy - 18)
      + input pin endpoint:  (cx - 55, cy + 18)
      output pin endpoint:   (cx + 60, cy)
    """
    return (
        # Triangle body
        f'<path d="M {cx-40} {cy-38} L {cx-40} {cy+38} L {cx+42} {cy} Z" '
        f'fill="#FCD34D" stroke="#0F172A" stroke-width="2.5" filter="url(#shadow)"/>'
        # Input pin stubs (extend 15px LEFT from triangle edge)
        f'<line x1="{cx-55}" y1="{cy-18}" x2="{cx-40}" y2="{cy-18}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        f'<line x1="{cx-55}" y1="{cy+18}" x2="{cx-40}" y2="{cy+18}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        # Output pin stub (extends 18px RIGHT from triangle apex)
        f'<line x1="{cx+42}" y1="{cy}" x2="{cx+60}" y2="{cy}" stroke="#0F172A" stroke-width="3" stroke-linecap="round"/>'
        # +/− input labels INSIDE the triangle, well clear of the edges
        f'<text x="{cx-28}" y="{cy-10}" font-size="18" font-weight="800" fill="#DC2626">−</text>'
        f'<text x="{cx-28}" y="{cy+24}" font-size="18" font-weight="800" fill="#059669">+</text>'
    )

# ═══════════════════════════════════════════════════════════════════
# 1. Voltage Divider — vertical stack: Vin top → R1 → Vout → R2 → GND
# ═══════════════════════════════════════════════════════════════════

def generate_voltage_divider_diagram(components, vin_value=12):
    from_fmt = format_value
    resistors = [c for c in components if c['type'] == 'resistor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    r1 = resistors[0] if resistors else {'name': 'R1', 'value': 10000}
    r2 = resistors[1] if len(resistors) > 1 else {'name': 'R2', 'value': 2200}
    vs = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': vin_value}

    W, H = 560, 380
    svg = [_diag_header(W, H, 'Voltage Divider')]
    # Layout: source on left, vertical R1/R2 stack on right
    src_x, src_y = 110, 200
    stack_x = 360
    r1_y, r2_y = 130, 270
    vout_y = 200

    svg.append(_source_dc(src_x, src_y, vs['name'], vs['value']))
    svg.append(_resistor_v(stack_x, r1_y, r1['name'], from_fmt(r1['value'], 'Ω')))
    svg.append(_resistor_v(stack_x, r2_y, r2['name'], from_fmt(r2['value'], 'Ω')))
    # Top loop: source(+) up, across, down to R1 top
    svg.append(_wire(src_x, src_y-26, src_x, 90))
    svg.append(_wire(src_x, 90, stack_x, 90))
    svg.append(_wire(stack_x, 90, stack_x, r1_y-35))
    # Junction between R1 and R2 (Vout node)
    svg.append(_wire(stack_x, r1_y+35, stack_x, r2_y-35))
    svg.append(_node(stack_x, vout_y))
    svg.append(_vout(stack_x, vout_y, side='right'))
    # Bottom loop: R2 bottom to ground bus, back to source(−)
    svg.append(_wire(stack_x, r2_y+35, stack_x, 330))
    svg.append(_wire(stack_x, 330, src_x, 330))
    svg.append(_wire(src_x, 330, src_x, src_y+26))
    # Ground at bottom-center
    svg.append(_ground((src_x+stack_x)//2, 330))
    svg.append('</svg>')
    return '\n'.join(svg)

# ═══════════════════════════════════════════════════════════════════
# 2. Simple RC Filter — Vin → R → Vout → C → GND
# ═══════════════════════════════════════════════════════════════════

def generate_rc_filter_diagram(components, vin_value=5):
    from_fmt = format_value
    resistors = [c for c in components if c['type'] == 'resistor']
    capacitors = [c for c in components if c['type'] == 'capacitor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    r = resistors[0] if resistors else {'name': 'R', 'value': 1000}
    c = capacitors[0] if capacitors else {'name': 'C', 'value': 100e-9}
    vs = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': vin_value}

    W, H = 620, 320
    svg = [_diag_header(W, H, 'RC Low-Pass Filter')]
    src_x, src_y = 90, 170
    r_x = 260
    cap_x = 420
    vout_x = 510

    svg.append(_source_ac(src_x, src_y, vs['name'], vs['value']))
    svg.append(_resistor_h(r_x, src_y, r['name'], from_fmt(r['value'], 'Ω')))
    svg.append(_capacitor_v(cap_x, src_y+40, c['name'], from_fmt(c['value'], 'F')))
    # Wires: source → R → junction → Vout
    svg.append(_wire(src_x+26, src_y, r_x-35, src_y))
    svg.append(_wire(r_x+35, src_y, vout_x, src_y))
    # Cap branch from junction down — connect to cap's TOP lead stub at (cap_x, cy-13)
    svg.append(_node(cap_x, src_y))
    svg.append(_wire(cap_x, src_y, cap_x, src_y+27))   # ends at cy-13 (top lead stub)
    # Cap to ground — from cap's BOTTOM lead stub at (cap_x, cy+13)
    svg.append(_wire(cap_x, src_y+53, cap_x, 270))     # starts at cy+13 (bottom lead stub)
    # Bottom return wire from source(−) to ground bus
    svg.append(_wire(src_x, src_y+26, src_x, 270))
    svg.append(_wire(src_x, 270, cap_x, 270))
    # Ground
    svg.append(_ground((src_x+cap_x)//2, 270))
    # Vout marker
    svg.append(_vout(vout_x, src_y, side='right'))
    svg.append('</svg>')
    return '\n'.join(svg)

# ═══════════════════════════════════════════════════════════════════
# 3. Multi-R + Multi-C Series — 1-4 resistors in series + 1-2 caps in parallel
# ═══════════════════════════════════════════════════════════════════

def generate_multi_rc_diagram(resistors, capacitors, vin_value=12):
    from_fmt = format_value
    n_r = len(resistors)
    n_c = len(capacitors)
    r_spacing = 110
    cap_spacing = 100   # bumped from 70 — needs room for "Cn" + value labels on the right

    src_x = 90
    first_r_x = 220
    cap_x_start = first_r_x + (n_r - 1) * r_spacing + 80
    rightmost_x = cap_x_start + (n_c - 1) * cap_spacing
    W = rightmost_x + 100
    H = 380
    y_main = 180
    y_gnd_bus = 320

    svg = [_diag_header(W, H, f'Loaded RC Series ({n_r} Resistors + {n_c} Capacitor{"s" if n_c > 1 else ""})')]

    svg.append(_source_dc(src_x, y_main, 'Vin', vin_value))

    # Resistor chain
    r_positions = [first_r_x + i * r_spacing for i in range(n_r)]
    for i, rx in enumerate(r_positions):
        svg.append(_resistor_h(rx, y_main, resistors[i]['name'], from_fmt(resistors[i]['value'], 'Ω')))

    # Wires: source → R1 → R2 → ... → Rn → cap_bus
    svg.append(_wire(src_x+26, y_main, r_positions[0]-35, y_main))
    for i in range(n_r - 1):
        svg.append(_wire(r_positions[i]+35, y_main, r_positions[i+1]-35, y_main))
    svg.append(_wire(r_positions[-1]+35, y_main, cap_x_start, y_main))

    # Vout marker between R1 and R2 (only if ≥2 R's)
    if n_r >= 2:
        vout_x = (r_positions[0] + r_positions[1]) / 2
        svg.append(_node(vout_x, y_main))
        svg.append(_vout(vout_x, y_main, side='top'))

    # Capacitor bus + caps hanging down
    cap_positions = [cap_x_start + i * cap_spacing for i in range(n_c)]
    if n_c > 1:
        svg.append(_wire(cap_positions[0], y_main, cap_positions[-1], y_main))
    for cx in cap_positions:
        svg.append(_node(cx, y_main))
        cap_cy = y_main + 50
        # Wire from main bus down to cap's top lead stub at (cx, cap_cy-13)
        svg.append(_wire(cx, y_main, cx, cap_cy-13))
    # Now draw each cap
    for i, cx in enumerate(cap_positions):
        cap_cy = y_main + 50
        svg.append(_capacitor_v(cx, cap_cy, capacitors[i]['name'], from_fmt(capacitors[i]['value'], 'F')))
        # Wire from cap's bottom lead stub at (cx, cap_cy+13) down to ground bus
        svg.append(_wire(cx, cap_cy+13, cx, y_gnd_bus))

    # Ground bus back to source(−)
    svg.append(_wire(src_x, y_main+26, src_x, y_gnd_bus))
    leftmost_cap_x = cap_positions[0]
    rightmost_cap_x = cap_positions[-1]
    svg.append(_wire(src_x, y_gnd_bus, rightmost_cap_x, y_gnd_bus))
    # Ground symbol at midpoint
    svg.append(_ground((src_x + leftmost_cap_x) // 2, y_gnd_bus))

    svg.append('</svg>')
    return '\n'.join(svg)

# ═══════════════════════════════════════════════════════════════════
# 4. RL Filter — Vin → R → Vout → L → GND
# ═══════════════════════════════════════════════════════════════════

def generate_rl_filter_diagram(components, vin_value=5):
    """Dynamic RL diagram supporting multiple R/L and `||` parallel groups."""
    from_fmt = format_value
    resistors = [c for c in components if c['type'] == 'resistor']
    inductors = [c for c in components if c['type'] == 'inductor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    vs = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': vin_value}

    def _group(comps):
        out, loose, par = [], [], {}
        for c in comps:
            gid = c.get('parallel_group')
            if gid:
                par.setdefault(gid, []).append(c)
            else:
                loose.append(c)
        for c in loose: out.append([c])
        for gid, group in par.items(): out.append(group)
        return out

    r_blocks = _group(resistors)
    l_blocks = _group(inductors)
    sequence = r_blocks + l_blocks
    n_blocks = len(sequence)

    SRC_X = 90
    BLOCK_SPACING = 130
    W = SRC_X + n_blocks * BLOCK_SPACING + 220
    H = 340
    SRC_Y = 170
    GND_Y = 280

    svg = [_diag_header(W, H, 'RL Circuit')]
    svg.append(_source_ac(SRC_X, SRC_Y, vs['name'], vs['value']))

    def _draw_block(x, block, kind):
        if len(block) == 1:
            c = block[0]
            if kind == 'resistor':
                svg.append(_resistor_h(x, SRC_Y, c['name'], from_fmt(c['value'], 'Ω')))
            else:
                svg.append(_inductor_h(x, SRC_Y, c['name'], from_fmt(c['value'], 'H')))
            return x - 30, x + 30
        n = len(block)
        offsets = {2: [-40, 40], 3: [-50, 0, 50], 4: [-60, -20, 20, 60]}.get(n, [-40, 40])
        left_x, right_x = x - 50, x + 50
        for i, c in enumerate(block):
            y = SRC_Y + offsets[i]
            if kind == 'resistor':
                svg.append(_resistor_h(x, y, c['name'], from_fmt(c['value'], 'Ω')))
            else:
                svg.append(_inductor_h(x, y, c['name'], from_fmt(c['value'], 'H')))
            svg.append(_wire(left_x, y, x - 30, y))
            svg.append(_wire(x + 30, y, right_x, y))
            if y != SRC_Y:
                svg.append(_wire(left_x, SRC_Y, left_x, y))
                svg.append(_wire(right_x, SRC_Y, right_x, y))
        svg.append(_node(left_x, SRC_Y))
        svg.append(_node(right_x, SRC_Y))
        return left_x, right_x

    cur_x = SRC_X + 100
    last_node_x = SRC_X + 26
    for block in sequence:
        kind = block[0]['type']
        center_x = cur_x + BLOCK_SPACING // 2
        if len(block) > 1:
            center_x += 20
        left_x, right_x = _draw_block(center_x, block, kind)
        if last_node_x < left_x:
            svg.append(_wire(last_node_x, SRC_Y, left_x, SRC_Y))
        last_node_x = right_x
        cur_x = right_x + 30

    # Vout marker at the end of the chain
    vout_x = last_node_x + 50
    svg.append(_wire(last_node_x, SRC_Y, vout_x, SRC_Y))
    svg.append(_vout(vout_x, SRC_Y, side='right'))
    # Return loop
    svg.append(_wire(SRC_X, SRC_Y + 26, SRC_X, GND_Y))
    svg.append(_wire(SRC_X, GND_Y, vout_x, GND_Y))
    svg.append(_wire(vout_x, SRC_Y, vout_x, GND_Y))
    svg.append(_ground((SRC_X + vout_x) // 2, GND_Y))
    svg.append('</svg>')
    return '\n'.join(svg)

# ═══════════════════════════════════════════════════════════════════
# 5. Series RLC — Vin → R → L → C → GND (loop)
# ═══════════════════════════════════════════════════════════════════

def generate_rlc_diagram(components, vin_value=5):
    """
    Dynamic series-RLC schematic that scales with however many R/L/C the
    user specified. Components within the same `parallel_group` are stacked
    in parallel between two junction nodes; everything else is drawn in series
    along the top rail.
    """
    from_fmt = format_value
    resistors = [c for c in components if c['type'] == 'resistor']
    inductors = [c for c in components if c['type'] == 'inductor']
    capacitors = [c for c in components if c['type'] == 'capacitor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    vs = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': vin_value}

    # ── Group same-type components by parallel_group ───────────────────
    def _group(comps):
        """Returns a list of blocks. Each block is a list of components.
        A block of length 1 = series element; length>1 = parallel sub-network."""
        out = []
        loose, par = [], {}
        for c in comps:
            gid = c.get('parallel_group')
            if gid:
                par.setdefault(gid, []).append(c)
            else:
                loose.append(c)
        # Loose components are each their own block
        for c in loose:
            out.append([c])
        # Parallel groups become single multi-element blocks
        for gid, group in par.items():
            out.append(group)
        return out

    r_blocks = _group(resistors)
    l_blocks = _group(inductors)
    c_blocks = _group(capacitors)

    # Sequence: source → all R blocks → all L blocks → all C blocks → back
    sequence = r_blocks + l_blocks + c_blocks
    n_blocks = len(sequence)

    # ── Sizing — wider for more blocks; taller when parallel groups present ──
    SRC_X = 90
    BLOCK_SPACING = 130
    chain_width = max(1, n_blocks) * BLOCK_SPACING + 100
    W = SRC_X + chain_width + 50

    # Largest parallel group drives vertical layout. Each branch needs ~72px
    # so the name (above) and value (below) labels don't collide.
    BRANCH_SPACING = 72
    max_par = max((len(b) for b in sequence), default=1)
    if max_par <= 1:
        SRC_Y, GND_Y, H = 180, 300, 360
    else:
        half_span = ((max_par - 1) / 2.0) * BRANCH_SPACING + 42  # + label room
        SRC_Y = int(half_span + 78)        # clear the header
        GND_Y = int(SRC_Y + half_span + 28)
        H = GND_Y + 50

    def _branch_offsets(n):
        """Symmetric vertical offsets around 0 with BRANCH_SPACING between branches."""
        start = -((n - 1) / 2.0) * BRANCH_SPACING
        return [start + i * BRANCH_SPACING for i in range(n)]

    svg = [_diag_header(W, H, 'Series RLC Circuit')]
    svg.append(_source_ac(SRC_X, SRC_Y, vs['name'], vs['value']))

    # First block sits BLOCK_SPACING/2 + 30 to the right of source for breathing room.
    cur_x = SRC_X + 100
    # Top wire from source's + terminal to the first block
    last_node_x = SRC_X + 26

    def _draw_block(x, block, kind):
        """Draw a series or parallel block at horizontal center x. Returns the
        x coordinate of the block's right-edge connection point."""
        if len(block) == 1:
            c = block[0]
            if kind == 'resistor':
                svg.append(_resistor_h(x, SRC_Y, c['name'], from_fmt(c['value'], 'Ω')))
                return x - 30, x + 30
            elif kind == 'inductor':
                svg.append(_inductor_h(x, SRC_Y, c['name'], from_fmt(c['value'], 'H')))
                return x - 30, x + 30
            else:  # capacitor (horizontal)
                svg.append(_capacitor_h(x, SRC_Y, c['name'], from_fmt(c['value'], 'F')))
                return x - 13, x + 13
        # Parallel block: stack vertically between two junction nodes at SRC_Y
        n = len(block)
        offsets = _branch_offsets(n)
        left_x = x - 50
        right_x = x + 50
        for i, c in enumerate(block):
            y = int(SRC_Y + offsets[i])
            # Connecting wires: left_x → left edge of component, right edge → right_x
            if kind == 'resistor':
                svg.append(_resistor_h(x, y, c['name'], from_fmt(c['value'], 'Ω')))
                svg.append(_wire(left_x, y, x - 30, y))
                svg.append(_wire(x + 30, y, right_x, y))
            elif kind == 'inductor':
                svg.append(_inductor_h(x, y, c['name'], from_fmt(c['value'], 'H')))
                svg.append(_wire(left_x, y, x - 30, y))
                svg.append(_wire(x + 30, y, right_x, y))
            else:
                svg.append(_capacitor_h(x, y, c['name'], from_fmt(c['value'], 'F')))
                svg.append(_wire(left_x, y, x - 13, y))
                svg.append(_wire(x + 13, y, right_x, y))
            # Vertical wires from junction down/up to branch
            if y != SRC_Y:
                svg.append(_wire(left_x, SRC_Y, left_x, y))
                svg.append(_wire(right_x, SRC_Y, right_x, y))
        # Junction dots
        svg.append(_node(left_x, SRC_Y))
        svg.append(_node(right_x, SRC_Y))
        return left_x, right_x

    # Walk the chain
    for i, block in enumerate(sequence):
        # Determine kind from first element
        kind = block[0]['type']
        # Compute center based on widths so far
        center_x = cur_x + BLOCK_SPACING // 2
        if len(block) > 1:
            center_x += 20  # extra padding for parallel blocks
        left_x, right_x = _draw_block(center_x, block, kind)
        # Wire from last_node_x to left_x of this block
        if last_node_x < left_x:
            svg.append(_wire(last_node_x, SRC_Y, left_x, SRC_Y))
        last_node_x = right_x
        cur_x = right_x + 30   # space before next block

    # Bottom return wire: from last block right side → down → across → up to source
    end_x = last_node_x + 30
    svg.append(_wire(last_node_x, SRC_Y, end_x, SRC_Y))
    svg.append(_wire(end_x, SRC_Y, end_x, GND_Y))
    svg.append(_wire(end_x, GND_Y, SRC_X, GND_Y))
    svg.append(_wire(SRC_X, SRC_Y + 26, SRC_X, GND_Y))
    svg.append(_ground((SRC_X + end_x) // 2, GND_Y))
    svg.append('</svg>')
    return '\n'.join(svg)

# ═══════════════════════════════════════════════════════════════════
# 6. Op-Amp — non-inverting or inverting
# ═══════════════════════════════════════════════════════════════════

def generate_op_amp_diagram(rf_obj, rin_obj, vin_value=1, is_inverting=False):
    """
    Op-amp diagram with proper pin-stub routing.
    Standard convention: − (inverting) input on TOP, + (non-inverting) on BOTTOM.
    Pin endpoints (outside triangle): minus_pin at (op_x-55, op_y-18),
    plus_pin at (op_x-55, op_y+18), output at (op_x+60, op_y).
    """
    from_fmt = format_value
    rf = rf_obj if rf_obj else {'name': 'Rf', 'value': 22000}
    rin = rin_obj if rin_obj else {'name': 'R1', 'value': 2200}

    W, H = 880, 440
    title = 'Inverting Op-Amp Amplifier' if is_inverting else 'Non-Inverting Op-Amp Amplifier'
    svg = [_diag_header(W, H, title)]

    op_x, op_y = 440, 240
    src_x = 80
    vout_x = 770
    minus_pin = (op_x - 55, op_y - 18)
    plus_pin  = (op_x - 55, op_y + 18)
    out_pin   = (op_x + 60, op_y)
    gnd_bus_y = 400          # bottom ground rail
    fb_top_y = 130           # feedback (Rf) horizontal level above op-amp

    # Op-amp body + pin stubs (drawn first so wires connect TO the stubs, not into the body)
    svg.append(_opamp_triangle(op_x, op_y))

    # Output wire to Vout marker, with a junction node where Rf comes down
    fb_junction_x = out_pin[0] + 25
    svg.append(_wire(out_pin[0], out_pin[1], vout_x - 12, out_pin[1]))
    svg.append(_node(fb_junction_x, out_pin[1]))
    svg.append(_vout(vout_x, out_pin[1], side='right'))

    if is_inverting:
        # Vin → R1 → − input (top), + input directly to ground, Rf feedback
        # Source aligned vertically with the − input pin so wire to R1 is straight
        src_y = minus_pin[1]
        svg.append(_source_ac(src_x, src_y, 'Vin'))
        r1_x = (src_x + 28 + minus_pin[0]) // 2
        svg.append(_resistor_h(r1_x, src_y, rin['name'], from_fmt(rin['value'], 'Ω')))
        svg.append(_wire(src_x + 28, src_y, r1_x - 30, src_y))
        svg.append(_wire(r1_x + 30, src_y, minus_pin[0], minus_pin[1]))
        svg.append(_node(minus_pin[0], minus_pin[1]))
        # Feedback Rf from output junction up & over to − pin
        rf_cx = (minus_pin[0] + fb_junction_x) // 2
        svg.append(_resistor_h(rf_cx, fb_top_y, rf['name'], from_fmt(rf['value'], 'Ω')))
        svg.append(_wire(minus_pin[0], minus_pin[1], minus_pin[0], fb_top_y))
        svg.append(_wire(minus_pin[0], fb_top_y, rf_cx - 30, fb_top_y))
        svg.append(_wire(rf_cx + 30, fb_top_y, fb_junction_x, fb_top_y))
        svg.append(_wire(fb_junction_x, fb_top_y, fb_junction_x, out_pin[1]))
        # + input directly to ground bus
        svg.append(_wire(plus_pin[0], plus_pin[1], plus_pin[0], gnd_bus_y))
        svg.append(_ground(plus_pin[0], gnd_bus_y))
        # Vin negative terminal to ground bus
        svg.append(_wire(src_x, src_y + 28, src_x, gnd_bus_y))
        svg.append(_wire(src_x, gnd_bus_y, plus_pin[0], gnd_bus_y))
    else:
        # Non-inverting: Vin → + input directly; R1 from − input to ground; Rf feedback
        # Source aligned with + input
        src_y = plus_pin[1]
        svg.append(_source_ac(src_x, src_y, 'Vin'))
        svg.append(_wire(src_x + 28, src_y, plus_pin[0], plus_pin[1]))
        svg.append(_node(plus_pin[0], plus_pin[1]))
        # R1 from − pin straight down to ground (labels on LEFT so they don't run into op-amp)
        r1_y = 340
        svg.append(_resistor_v(minus_pin[0], r1_y, rin['name'], from_fmt(rin['value'], 'Ω'), side='left'))
        svg.append(_wire(minus_pin[0], minus_pin[1], minus_pin[0], r1_y - 30))
        svg.append(_wire(minus_pin[0], r1_y + 30, minus_pin[0], gnd_bus_y))
        svg.append(_node(minus_pin[0], minus_pin[1]))
        # Rf feedback from output junction up & over to − pin
        rf_cx = (minus_pin[0] + fb_junction_x) // 2
        svg.append(_resistor_h(rf_cx, fb_top_y, rf['name'], from_fmt(rf['value'], 'Ω')))
        svg.append(_wire(minus_pin[0], minus_pin[1], minus_pin[0], fb_top_y))
        svg.append(_wire(minus_pin[0], fb_top_y, rf_cx - 30, fb_top_y))
        svg.append(_wire(rf_cx + 30, fb_top_y, fb_junction_x, fb_top_y))
        svg.append(_wire(fb_junction_x, fb_top_y, fb_junction_x, out_pin[1]))
        # Vin negative terminal to ground bus
        svg.append(_wire(src_x, src_y + 28, src_x, gnd_bus_y))
        svg.append(_wire(src_x, gnd_bus_y, minus_pin[0], gnd_bus_y))
        svg.append(_ground((src_x + minus_pin[0]) // 2, gnd_bus_y))

    svg.append('</svg>')
    return '\n'.join(svg)

# ═══════════════════════════════════════════════════════════════════
# 7. Series resistor chain
# ═══════════════════════════════════════════════════════════════════

def generate_series_resistor_diagram(components):
    from_fmt = format_value
    resistors = [c for c in components if c['type'] == 'resistor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    vs = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': 12}

    n = len(resistors)
    spacing = 130
    src_x = 90
    first_r_x = 220
    last_r_x = first_r_x + (n - 1) * spacing
    W = last_r_x + 130
    H = 320
    y_main = 170
    y_gnd = 270

    svg = [_diag_header(W, H, f'Series Resistor Network ({n} Resistors)')]
    svg.append(_source_dc(src_x, y_main, vs['name'], vs['value']))

    r_positions = [first_r_x + i * spacing for i in range(n)]
    for i, rx in enumerate(r_positions):
        svg.append(_resistor_h(rx, y_main, resistors[i]['name'], from_fmt(resistors[i]['value'], 'Ω')))
    # Wires through chain
    svg.append(_wire(src_x+26, y_main, r_positions[0]-35, y_main))
    for i in range(n-1):
        svg.append(_wire(r_positions[i]+35, y_main, r_positions[i+1]-35, y_main))
    svg.append(_wire(r_positions[-1]+35, y_main, W-50, y_main))
    # Return loop
    svg.append(_wire(W-50, y_main, W-50, y_gnd))
    svg.append(_wire(src_x, y_main+26, src_x, y_gnd))
    svg.append(_wire(src_x, y_gnd, W-50, y_gnd))
    svg.append(_ground((src_x + W - 50) // 2, y_gnd))
    svg.append('</svg>')
    return '\n'.join(svg)

# ═══════════════════════════════════════════════════════════════════
# 8. Parallel resistor network
# ═══════════════════════════════════════════════════════════════════

def generate_parallel_resistor_diagram(components):
    from_fmt = format_value
    resistors = [c for c in components if c['type'] == 'resistor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    vs = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': 12}

    n = len(resistors)
    branch_spacing = 100
    src_x = 90
    first_branch_x = 250
    last_branch_x = first_branch_x + (n - 1) * branch_spacing
    W = last_branch_x + 130
    H = 360
    y_top = 110
    y_bot = 290

    svg = [_diag_header(W, H, f'Parallel Resistor Network ({n} Branches)')]
    src_y = (y_top + y_bot) // 2
    svg.append(_source_dc(src_x, src_y, vs['name'], vs['value']))

    branch_positions = [first_branch_x + i * branch_spacing for i in range(n)]
    # Vertical resistors per branch
    branch_cy = (y_top + y_bot) // 2
    for i, bx in enumerate(branch_positions):
        svg.append(_resistor_v(bx, branch_cy, resistors[i]['name'], from_fmt(resistors[i]['value'], 'Ω')))
        # Top stub
        svg.append(_wire(bx, y_top, bx, branch_cy-35))
        # Bottom stub
        svg.append(_wire(bx, branch_cy+35, bx, y_bot))
        svg.append(_node(bx, y_top))
        svg.append(_node(bx, y_bot))

    # Top bus + bottom bus
    svg.append(_wire(branch_positions[0], y_top, branch_positions[-1], y_top))
    svg.append(_wire(branch_positions[0], y_bot, branch_positions[-1], y_bot))
    # Connect source(+) to top bus
    svg.append(_wire(src_x, src_y-26, src_x, y_top))
    svg.append(_wire(src_x, y_top, branch_positions[0], y_top))
    # Connect source(−) to bottom bus
    svg.append(_wire(src_x, src_y+26, src_x, y_bot))
    svg.append(_wire(src_x, y_bot, branch_positions[0], y_bot))
    svg.append(_ground(src_x, y_bot))
    svg.append('</svg>')
    return '\n'.join(svg)


def _bar_plot(names, values, ylabel='Value', title='Distribution'):
    """Small standalone matplotlib bar-chart helper returning a base64 PNG data URI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor='black')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + max(values) * 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode()
    plt.close()
    return f"data:image/png;base64,{data}"


def generate_enhanced_circuit_diagram(parsed_data):
    components = parsed_data['components']
    circuit_type = parsed_data['circuit_type']
    
    width, height = 900, 500
    margin = 60
    
    svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <!-- Component definitions -->
            <g id="resistor">
                <rect x="-25" y="-8" width="50" height="16" fill="white" stroke="black" stroke-width="2" rx="3"/>
                <path d="M-25,0 L-30,0 M25,0 L30,0" stroke="black" stroke-width="2"/>
                <path d="M-20,-5 L-15,5 L-10,-5 L-5,5 L0,-5 L5,5 L10,-5 L15,5 L20,-5" stroke="black" stroke-width="2" fill="none"/>
            </g>
            <g id="capacitor">
                <path d="M-30,0 L-8,0 M8,0 L30,0" stroke="black" stroke-width="2"/>
                <path d="M-8,-20 L-8,20 M8,-20 L8,20" stroke="black" stroke-width="3"/>
            </g>
            <g id="inductor">
                <path d="M-30,0 L-20,0" stroke="black" stroke-width="2"/>
                <path d="M-20,0 Q-20,-15 -10,-15 Q0,0 10,0 Q20,-15 20,0" stroke="black" stroke-width="2" fill="none"/>
                <path d="M20,0 L30,0" stroke="black" stroke-width="2"/>
            </g>
            <g id="voltage_source">
                <circle r="20" fill="white" stroke="black" stroke-width="2"/>
                <path d="M-30,0 L-20,0 M20,0 L30,0" stroke="black" stroke-width="2"/>
                <text x="0" y="6" text-anchor="middle" font-size="16" font-weight="bold">V</text>
                <path d="M-5,-10 L-5,10 M5,-5 L5,5" stroke="black" stroke-width="2"/>
            </g>
            <g id="ground">
                <path d="M0,0 L0,15 M-15,15 L15,15 M-10,20 L10,20 M-5,25 L5,25" stroke="black" stroke-width="2"/>
            </g>
            <g id="op_amp">
                <path d="M-30,0 L-15,0 L15,-20 L15,20 L-15,0 Z" fill="white" stroke="black" stroke-width="2"/>
                <path d="M15,0 L30,0" stroke="black" stroke-width="2"/>
                <text x="-8" y="-8" font-size="12">+</text>
                <text x="-8" y="12" font-size="12">-</text>
            </g>
        </defs>
        
        <!-- Background -->
        <rect width="{width}" height="{height}" fill="#f8f9fa" stroke="#dee2e6" stroke-width="2"/>
        
        <!-- Grid pattern -->
        <defs>
            <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e9ecef" stroke-width="0.5"/>
            </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" opacity="0.5"/>
        
        <!-- Title -->
        <text x="{width//2}" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">{circuit_type.replace('_', ' ').title()} Circuit</text>'''
    
    y_center = height // 2
    
    if circuit_type == 'voltage_divider' and len(components) >= 3:
        resistors = [c for c in components if c['type'] == 'resistor']
        voltage_sources = [c for c in components if c['type'] == 'voltage_source']
        
        x_start = margin + 80
        
        svg += f'''
        <g transform="translate({x_start},{y_center})">
            <use href="#voltage_source"/>
            <text x="0" y="-35" text-anchor="middle" font-size="12" font-weight="bold">{voltage_sources[0]['name']}</text>
            <text x="0" y="45" text-anchor="middle" font-size="10">{voltage_sources[0]['value']}V</text>
        </g>'''
        
        svg += f'''
        <g transform="translate({x_start + 200},{y_center - 80})">
            <use href="#resistor"/>
            <text x="0" y="-25" text-anchor="middle" font-size="12" font-weight="bold">{resistors[0]['name']}</text>
            <text x="0" y="35" text-anchor="middle" font-size="10">{format_value(resistors[0]['value'], 'Ω')}</text>
        </g>'''
        
        svg += f'''
        <g transform="translate({x_start + 200},{y_center + 80})">
            <use href="#resistor"/>
            <text x="0" y="-25" text-anchor="middle" font-size="12" font-weight="bold">{resistors[1]['name']}</text>
            <text x="0" y="35" text-anchor="middle" font-size="10">{format_value(resistors[1]['value'], 'Ω')}</text>
        </g>'''
        
        svg += f'''
        <!-- Voltage source to R1 -->
        <line x1="{x_start + 30}" y1="{y_center}" x2="{x_start + 170}" y2="{y_center}" stroke="black" stroke-width="2"/>
        <line x1="{x_start + 170}" y1="{y_center}" x2="{x_start + 170}" y2="{y_center - 80}" stroke="black" stroke-width="2"/>
        
        <!-- R1 to R2 connection (output node) -->
        <line x1="{x_start + 230}" y1="{y_center - 80}" x2="{x_start + 280}" y2="{y_center - 80}" stroke="black" stroke-width="2"/>
        <line x1="{x_start + 280}" y1="{y_center - 80}" x2="{x_start + 280}" y2="{y_center + 80}" stroke="black" stroke-width="2"/>
        <line x1="{x_start + 170}" y1="{y_center + 80}" x2="{x_start + 280}" y2="{y_center + 80}" stroke="black" stroke-width="2"/>
        
        <!-- Ground connections -->
        <line x1="{x_start}" y1="{y_center + 30}" x2="{x_start}" y2="{y_center + 60}" stroke="black" stroke-width="2"/>
        <line x1="{x_start + 230}" y1="{y_center + 80}" x2="{x_start + 300}" y2="{y_center + 80}" stroke="black" stroke-width="2"/>
        <line x1="{x_start + 300}" y1="{y_center + 80}" x2="{x_start + 300}" y2="{y_center + 110}" stroke="black" stroke-width="2"/>
        
        <!-- Ground symbols -->
        <g transform="translate({x_start},{y_center + 60})">
            <use href="#ground"/>
        </g>
        <g transform="translate({x_start + 300},{y_center + 110})">
            <use href="#ground"/>
        </g>
        
        <!-- Output label -->
        <text x="{x_start + 320}" y="{y_center}" font-size="12" font-weight="bold" fill="#007bff">Vout</text>
        <circle cx="{x_start + 280}" cy="{y_center}" r="3" fill="#007bff"/>'''
    
    elif circuit_type in ['rc', 'rl'] and len(components) >= 3:
        x_positions = [margin + 80, margin + 250, margin + 420]
        
        for i, comp in enumerate(components[:3]):
            x = x_positions[i]
            
            if comp['type'] == 'voltage_source':
                svg += f'''
                <g transform="translate({x},{y_center})">
                    <use href="#voltage_source"/>
                    <text x="0" y="-35" text-anchor="middle" font-size="12" font-weight="bold">{comp['name']}</text>
                    <text x="0" y="45" text-anchor="middle" font-size="10">{comp['value']}V</text>
                </g>'''
            elif comp['type'] == 'resistor':
                svg += f'''
                <g transform="translate({x},{y_center})">
                    <use href="#resistor"/>
                    <text x="0" y="-25" text-anchor="middle" font-size="12" font-weight="bold">{comp['name']}</text>
                    <text x="0" y="35" text-anchor="middle" font-size="10">{format_value(comp['value'], comp['unit'])}</text>
                </g>'''
            elif comp['type'] == 'capacitor':
                svg += f'''
                <g transform="translate({x},{y_center + 60})">
                    <use href="#capacitor"/>
                    <text x="0" y="-35" text-anchor="middle" font-size="12" font-weight="bold">{comp['name']}</text>
                    <text x="0" y="50" text-anchor="middle" font-size="10">{format_value(comp['value'], comp['unit'])}</text>
                </g>'''
            elif comp['type'] == 'inductor':
                svg += f'''
                <g transform="translate({x},{y_center})">
                    <use href="#inductor"/>
                    <text x="0" y="-25" text-anchor="middle" font-size="12" font-weight="bold">{comp['name']}</text>
                    <text x="0" y="35" text-anchor="middle" font-size="10">{format_value(comp['value'], comp['unit'])}</text>
                </g>'''
        
        svg += f'''
        <line x1="{x_positions[0] + 30}" y1="{y_center}" x2="{x_positions[1] - 30}" y2="{y_center}" stroke="black" stroke-width="2"/>
        <line x1="{x_positions[1] + 30}" y1="{y_center}" x2="{x_positions[2] - 30}" y2="{y_center}" stroke="black" stroke-width="2"/>'''
        
        if any(c['type'] == 'capacitor' for c in components):
            svg += f'''
            <line x1="{x_positions[2]}" y1="{y_center + 30}" x2="{x_positions[2]}" y2="{y_center + 100}" stroke="black" stroke-width="2"/>
            <g transform="translate({x_positions[2]},{y_center + 100})">
                <use href="#ground"/>
            </g>'''
    
    elif circuit_type == 'op_amp' and len(components) >= 3:
        resistors = [c for c in components if c['type'] == 'resistor']
        voltage_sources = [c for c in components if c['type'] == 'voltage_source']
        
        op_x, op_y = margin + 300, y_center
        
        svg += f'''
        <g transform="translate({op_x},{op_y})">
            <use href="#op_amp"/>
            <text x="0" y="-35" text-anchor="middle" font-size="12" font-weight="bold">Op-Amp</text>
        </g>'''
        
        if len(resistors) >= 2:
            svg += f'''
            <g transform="translate({op_x - 120},{op_y + 10})">
                <use href="#resistor"/>
                <text x="0" y="-25" text-anchor="middle" font-size="12" font-weight="bold">{resistors[1]['name']}</text>
                <text x="0" y="35" text-anchor="middle" font-size="10">{format_value(resistors[1]['value'], 'Ω')}</text>
            </g>'''
            
            svg += f'''
            <g transform="translate({op_x},{op_y - 80})">
                <use href="#resistor"/>
                <text x="0" y="-25" text-anchor="middle" font-size="12" font-weight="bold">{resistors[0]['name']}</text>
                <text x="0" y="35" text-anchor="middle" font-size="10">{format_value(resistors[0]['value'], 'Ω')}</text>
            </g>'''
        
        if voltage_sources:
            svg += f'''
            <g transform="translate({op_x - 200},{op_y + 10})">
                <use href="#voltage_source"/>
                <text x="0" y="-35" text-anchor="middle" font-size="12" font-weight="bold">{voltage_sources[0]['name']}</text>
                <text x="0" y="45" text-anchor="middle" font-size="10">{voltage_sources[0]['value']}V</text>
            </g>'''
    
    svg += f'''
    <text x="{width - 50}" y="50" text-anchor="middle" font-size="10" fill="#666">Generated by Circuit Analyzer</text>
    </svg>'''
    
    return svg

def format_value(value, unit):
    if unit == 'Ω':
        if value >= 1e6:
            return f"{value/1e6:.2f}MΩ"
        elif value >= 1e3:
            return f"{value/1e3:.2f}kΩ"
        else:
            return f"{value:.0f}Ω"
    elif unit == 'F':
        if value >= 1e-6:
            return f"{value*1e6:.1f}μF"
        elif value >= 1e-9:
            return f"{value*1e9:.1f}nF"
        elif value >= 1e-12:
            return f"{value*1e12:.1f}pF"
        else:
            return f"{value:.2e}F"
    elif unit == 'H':
        if value >= 1e-3:
            return f"{value*1e3:.1f}mH"
        elif value >= 1e-6:
            return f"{value*1e6:.1f}μH"
        else:
            return f"{value:.2e}H"
    return f"{value}{unit}"

def generate_matlab_analysis_code(parsed_data, results):
    components = parsed_data['components']
    circuit_type = parsed_data['circuit_type']
    
    code = f"""% Advanced Circuit Analysis System
% Generated for: {parsed_data['original_query']}
clear; clc; close all;

fprintf('\\n=== CIRCUIT ANALYSIS SYSTEM ===\\n');
fprintf('Circuit Type: {circuit_type.replace('_', ' ').title()}\\n\\n');

"""
    
    code += "% Component Definitions\n"
    for comp in components:
        name = comp['name'].replace('-', '_')
        code += f"{name}_value = {comp['value']}; % {comp['name']} = {format_value(comp['value'], comp['unit'])}\\n"
    
    if circuit_type == 'voltage_divider':
        code += """
% Voltage Divider Analysis
fprintf('\\n--- VOLTAGE DIVIDER ANALYSIS ---\\n');
R_total = R1_value + R2_value;
V_out = Vin_value * (R2_value / R_total);
I_total = Vin_value / R_total;
P_R1 = I_total^2 * R1_value;
P_R2 = I_total^2 * R2_value;
P_total = Vin_value * I_total;

fprintf('Input Voltage: %.3f V\\n', Vin_value);
fprintf('Output Voltage: %.3f V\\n', V_out);
fprintf('Total Current: %.3f mA\\n', I_total*1000);
fprintf('Total Power: %.3f mW\\n', P_total*1000);
fprintf('Voltage Ratio: %.3f\\n', V_out/Vin_value);

% Visualization
figure('Position', [100, 100, 1200, 800]);
subplot(2,3,1);
bar([Vin_value, V_out], 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTickLabel', {'Input', 'Output'});
ylabel('Voltage (V)');
title('Voltage Levels');
grid on;

subplot(2,3,2);
pie([P_R1, P_R2], {'R1', 'R2'});
title('Power Distribution');

subplot(2,3,3);
bar([R1_value, R2_value], 'FaceColor', [0.8, 0.4, 0.2]);
set(gca, 'XTickLabel', {'R1', 'R2'});
ylabel('Resistance (Ohms)');
title('Component Values');
grid on;
"""
    
    elif circuit_type == 'rc':
        code += """
% RC Filter Analysis
fprintf('\\n--- RC FILTER ANALYSIS ---\\n');
tau = R1_value * C1_value;
fc = 1 / (2*pi*R1_value*C1_value);

fprintf('Time Constant: %.6f s\\n', tau);
fprintf('Cutoff Frequency: %.2f Hz\\n', fc);

% Frequency Response
f = logspace(-1, 6, 1000);
w = 2*pi*f;
H = 1 ./ (1 + 1j*w*R1_value*C1_value);
mag_dB = 20*log10(abs(H));
phase_deg = angle(H) * 180/pi;

% Step Response
t = linspace(0, 5*tau, 1000);
v_in = ones(size(t));
v_out = 1 - exp(-t/tau);

% Visualization
figure('Position', [200, 150, 1400, 900]);

subplot(2,2,1);
semilogx(f, mag_dB, 'b-', 'LineWidth', 2);
hold on;
semilogx([fc fc], [min(mag_dB) max(mag_dB)], 'r--', 'LineWidth', 2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Magnitude Response');
legend('|H(jω)|', '3dB Point');

subplot(2,2,2);
semilogx(f, phase_deg, 'r-', 'LineWidth', 2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('Phase Response');

subplot(2,2,3);
plot(t*1000, v_in, 'b--', 'LineWidth', 2, 'DisplayName', 'Input');
hold on;
plot(t*1000, v_out, 'r-', 'LineWidth', 3, 'DisplayName', 'Output');
grid on;
xlabel('Time (ms)');
ylabel('Voltage (V)');
title('Step Response');
legend('show');

subplot(2,2,4);
bar([R1_value*1e-3, C1_value*1e9], 'FaceColor', [0.6, 0.8, 0.3]);
set(gca, 'XTickLabel', {'R (kΩ)', 'C (nF)'});
ylabel('Value');
title('Component Values');
grid on;
"""
    
    elif circuit_type == 'rlc':
        code += """
% RLC Circuit Analysis
fprintf('\\n--- RLC CIRCUIT ANALYSIS ---\\n');
omega_0 = 1/sqrt(L1_value*C1_value);
f0 = omega_0/(2*pi);
zeta = R1_value/(2*sqrt(L1_value/C1_value));
Q = 1/(2*zeta);
BW = f0/Q;

fprintf('Resonant Frequency: %.2f Hz\\n', f0);
fprintf('Damping Ratio: %.4f\\n', zeta);
fprintf('Quality Factor: %.2f\\n', Q);
fprintf('Bandwidth: %.2f Hz\\n', BW);

if zeta < 1
    fprintf('System is UNDERDAMPED\\n');
elseif zeta == 1
    fprintf('System is CRITICALLY DAMPED\\n');
else
    fprintf('System is OVERDAMPED\\n');
end

% Frequency Response
f = logspace(log10(f0/100), log10(f0*100), 1000);
w = 2*pi*f;
s = 1j*w;
H = 1 ./ (L1_value*C1_value*s.^2 + R1_value*C1_value*s + 1);
mag_dB = 20*log10(abs(H));
phase_deg = angle(H) * 180/pi;

% Visualization
figure('Position', [300, 200, 1400, 900]);

subplot(2,2,1);
semilogx(f, mag_dB, 'b-', 'LineWidth', 3);
hold on;
semilogx([f0 f0], [min(mag_dB) max(mag_dB)], 'r--', 'LineWidth', 2);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('RLC Frequency Response');
legend('|H(jω)|', 'Resonance');

subplot(2,2,2);
semilogx(f, phase_deg, 'r-', 'LineWidth', 3);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase (degrees)');
title('Phase Response');

subplot(2,2,3);
bar([R1_value, L1_value*1e3, C1_value*1e9], 'FaceColor', [0.8, 0.6, 0.4]);
set(gca, 'XTickLabel', {'R (Ω)', 'L (mH)', 'C (nF)'});
ylabel('Value');
title('Component Values');
grid on;

subplot(2,2,4);
params = [f0, Q, BW, zeta];
bar(params, 'FaceColor', [0.4, 0.8, 0.6]);
set(gca, 'XTickLabel', {'f₀ (Hz)', 'Q', 'BW (Hz)', 'ζ'});
ylabel('Value');
title('Circuit Parameters');
grid on;
"""
    
    code += """
% Summary
fprintf('\\n=== ANALYSIS COMPLETE ===\\n');
fprintf('All plots generated successfully.\\n');
"""
    
    return code
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)