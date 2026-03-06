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
        "backend_ready": True
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
def parse_circuit_query(query):
    components = []
    calculations_requested = []
    query_lower = query.lower()
    
    logger.info(f"Parsing query: '{query}'")
    
    calc_requests = {
        'current': r'(?:find|calculate|what.*is|determine|compute).*(?:current|amperage|amps?)',
        'voltage': r'(?:find|calculate|what.*is|determine|compute).*(?:voltage|output|potential|volts?)',
        'power': r'(?:find|calculate|what.*is|determine|compute).*(?:power|watts?|dissipation)',
        'resistance': r'(?:find|calculate|what.*is|determine|compute).*(?:resistance|impedance|ohms?)',
        'frequency': r'(?:find|calculate|what.*is|determine|compute).*(?:frequency|cutoff|resonant)',
        'gain': r'(?:find|calculate|what.*is|determine|compute).*(?:gain|amplification)',
        'impedance': r'(?:find|calculate|what.*is|determine|compute).*(?:impedance|reactance)'
    }
    
    for calc_type, pattern in calc_requests.items():
        if re.search(pattern, query_lower):
            calculations_requested.append(calc_type)
    
    resistor_patterns = [
        r'r(\d+)\s*=\s*(\d+(?:\.\d+)?)\s*([kmg]?)\s*(?:ohm|Ω)',
        r'r(\d+)\s*=\s*(\d+(?:\.\d+)?)([kmg]?)',
        r'(\d+(?:\.\d+)?)\s*([kmg]?)\s*ohm\s+resistor',
        r'resistor.*?(\d+(?:\.\d+)?)\s*([kmg]?)\s*(?:ohm|Ω)',
    ]
    
    resistor_count = 1
    for pattern in resistor_patterns:
        logger.info(f"Trying resistor pattern: {pattern}")
        matches = list(re.finditer(pattern, query, re.IGNORECASE))
        logger.info(f"Found {len(matches)} matches")
        
        for match in matches:
            logger.info(f"Match groups: {match.groups()}")
            groups = match.groups()
            
            if len(groups) >= 2:
                value_str = groups[-2]
                prefix = groups[-1] if groups[-1] else None
                
                value = parse_value(value_str, prefix)
                logger.info(f"Parsed resistor: {value} ohms")
                
                components.append({
                    'type': 'resistor',
                    'name': f'R{resistor_count}',
                    'value': value,
                    'unit': 'Ω'
                })
                resistor_count += 1
    
    voltage_patterns = [
        r'(?:input|voltage).*?(\d+(?:\.\d+)?)\s*v',
        r'(\d+(?:\.\d+)?)\s*v',
    ]
    
    for pattern in voltage_patterns:
        logger.info(f"Trying voltage pattern: {pattern}")
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            logger.info(f"Voltage match: {match.groups()}")
            value = float(match.group(1))
            components.append({
                'type': 'voltage_source',
                'name': 'Vin',
                'value': value,
                'unit': 'V'
            })
            break
    
    logger.info(f"Found components: {components}")
    
    circuit_type = determine_circuit_type(query_lower, components)
    logger.info(f"Determined circuit type: {circuit_type}")
    
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
    logger.info(f"Query contains 'series': {'series' in query_lower}")
    
    if any(keyword in query_lower for keyword in ['voltage divider', 'potential divider']):
        return 'voltage_divider'
    elif any(keyword in query_lower for keyword in ['op-amp', 'operational amplifier', 'op amp']):
        return 'op_amp'
    elif any(keyword in query_lower for keyword in ['filter', 'low pass', 'high pass', 'band pass']):
        return 'filter'
    elif any(keyword in query_lower for keyword in ['rlc', 'resonant', 'resonance', 'tank']):
        return 'rlc'
    
    elif 'series' in query_lower:
        if len(resistors) >= 2:
            return 'series_resistors'
        elif len(resistors) >= 1 and len(capacitors) >= 1:
            return 'rc_series'
        elif len(resistors) >= 1 and len(inductors) >= 1:
            return 'rl_series'
        elif len(resistors) >= 1 and len(capacitors) >= 1 and len(inductors) >= 1:
            return 'rlc_series'
    elif 'parallel' in query_lower:
        if len(resistors) >= 2:
            return 'parallel_resistors'
        elif len(resistors) >= 1 and len(capacitors) >= 1:
            return 'rc_parallel'
    
    elif len(resistors) == 2 and len(voltage_sources) >= 1 and not capacitors and not inductors:
        if 'output' in query_lower and 'voltage' in query_lower:
            return 'voltage_divider'
        else:
            return 'series_resistors'
    elif len(resistors) >= 2 and len(voltage_sources) >= 1 and not capacitors and not inductors:
        return 'series_resistors'
    elif len(resistors) >= 1 and len(capacitors) >= 1 and not inductors:
        return 'rc_circuit'
    elif len(resistors) >= 1 and len(inductors) >= 1 and not capacitors:
        return 'rl_circuit'
    elif len(resistors) >= 1 and len(capacitors) >= 1 and len(inductors) >= 1:
        return 'rlc_circuit'
    elif len(resistors) >= 2 and any(keyword in query_lower for keyword in ['feedback', 'amplifier']):
        return 'op_amp'
    
    logger.warning(f"Could not determine circuit type, returning 'series_resistors' as default")
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
    
    resistors = [c for c in components if c['type'] == 'resistor']
    capacitors = [c for c in components if c['type'] == 'capacitor']
    inductors = [c for c in components if c['type'] == 'inductor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    
    vin = voltage_sources[0]['value'] if voltage_sources else 5.0
    
    if circuit_type in ['voltage_divider']:
        return analyze_voltage_divider(components, calc_requests, options, vin)
    elif circuit_type in ['series_resistors', 'resistor_network']:
        return analyze_series_resistors(components, calc_requests, options, vin)
    elif circuit_type in ['parallel_resistors']:
        return analyze_parallel_resistors(components, calc_requests, options, vin)
    elif circuit_type in ['rc_circuit', 'rc_series']:
        return analyze_rc_circuit(components, calc_requests, options, vin)
    elif circuit_type in ['rl_circuit', 'rl_series']:
        return analyze_rl_circuit(components, calc_requests, options, vin)
    elif circuit_type in ['rlc_circuit', 'rlc_series', 'rlc']:
        return analyze_rlc_circuit(components, calc_requests, options, vin)
    elif circuit_type == 'op_amp':
        return analyze_op_amp(components, calc_requests, options, vin)
    else:
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
    
    r_total = sum(r['value'] for r in resistors)
    current = vin / r_total
    
    voltages = [(current * r['value'], r['name']) for r in resistors]
    powers = [(current**2 * r['value'], r['name']) for r in resistors]
    total_power = sum(p[0] for p in powers)
    
    results['calculations']['series_analysis'] = {
        'total_resistance': r_total,
        'current': current,
        'total_power': total_power,
        'individual_voltages': voltages,
        'individual_powers': powers
    }
    
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
    
    if options.get('generate_plots', False):
        results['plots'].append({
            'title': 'Voltage Distribution',
            'description': 'Voltage across each resistor in series',
            'image': generate_voltage_distribution_plot(voltages)
        })
        
        results['plots'].append({
            'title': 'Power Distribution',
            'description': 'Power dissipated by each resistor',
            'image': generate_power_distribution_plot([{'name': name, 'power': power} for power, name in powers])
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
    
    r1, r2 = resistors[0]['value'], resistors[1]['value']
    vd_result = calculator.calculate_voltage_divider(r1, r2, vin)
    results['calculations']['voltage_divider'] = vd_result
    
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
    
    return results


def analyze_rc_circuit(components, calc_requests, options, vin):
    results = {
        'direct_answers': [],
        'calculations': {},
        'plots': [],
        'circuit_diagram': None
    }
    
    resistors = [c for c in components if c['type'] == 'resistor']
    capacitors = [c for c in components if c['type'] == 'capacitor']
    
    if not resistors or not capacitors:
        return results
    
    r = resistors[0]['value']
    c = capacitors[0]['value']
    rc_result = calculator.calculate_rc_circuit(r, c)
    results['calculations']['rc_filter'] = rc_result
    
    results['direct_answers'].append({
        'question': 'Cutoff Frequency',
        'answer': f'{rc_result["cutoff_frequency"]:.2f} Hz'
    })
    
    return results

def analyze_rl_circuit(components, calc_requests, options, vin):
    return {'direct_answers': [], 'calculations': {}, 'plots': [], 'circuit_diagram': None}

def analyze_rlc_circuit(components, calc_requests, options, vin):
    return {'direct_answers': [], 'calculations': {}, 'plots': [], 'circuit_diagram': None}

def analyze_op_amp(components, calc_requests, options, vin):
    return {'direct_answers': [], 'calculations': {}, 'plots': [], 'circuit_diagram': None}


def generate_series_resistor_diagram(components):
    resistors = [c for c in components if c['type'] == 'resistor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    
    vin = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': 12}
    
    svg = f'''<svg width="700" height="300" viewBox="0 0 700 300">
        <rect width="100%" height="100%" fill="#f8f9fa"/>
        <text x="350" y="25" text-anchor="middle" font-size="16" font-weight="bold">Series Resistor Circuit</text>
        
        <!-- Voltage Source -->
        <circle cx="80" cy="150" r="25" fill="white" stroke="black" stroke-width="2"/>
        <text x="80" y="145" text-anchor="middle" font-size="16" font-weight="bold">+</text>
        <text x="80" y="165" text-anchor="middle" font-size="16" font-weight="bold">-</text>
        <text x="80" y="190" text-anchor="middle" font-size="12" font-weight="bold">{vin['name']}</text>
        <text x="80" y="205" text-anchor="middle" font-size="10">{vin['value']}V</text>
        
        <!-- Resistors in series -->'''
    
    x_positions = [150 + i * 120 for i in range(len(resistors))]
    colors = ['#FF6B35', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    
    for i, (resistor, x) in enumerate(zip(resistors, x_positions)):
        svg += f'''
        <rect x="{x-30}" y="140" width="60" height="20" fill="{colors[i % len(colors)]}" stroke="black" stroke-width="2" rx="3"/>
        <text x="{x}" y="130" text-anchor="middle" font-size="12" font-weight="bold">{resistor['name']}</text>
        <text x="{x}" y="180" text-anchor="middle" font-size="10">{format_value(resistor['value'], 'Ω')}</text>'''
    
    svg += f'''
        <!-- Top connections -->
        <line x1="80" y1="125" x2="80" y2="100" stroke="black" stroke-width="3"/>
        <line x1="80" y1="100" x2="620" y2="100" stroke="black" stroke-width="3"/>
        <line x1="620" y1="100" x2="620" y2="150" stroke="black" stroke-width="3"/>'''
    
    for i in range(len(x_positions)):
        x = x_positions[i]
        svg += f'<line x1="{x-30}" y1="100" x2="{x-30}" y2="150" stroke="black" stroke-width="2"/>'
        svg += f'<line x1="{x+30}" y1="150" x2="{x+30}" y2="100" stroke="black" stroke-width="2"/>'
        
        if i < len(x_positions) - 1:
            next_x = x_positions[i + 1]
            svg += f'<line x1="{x+30}" y1="150" x2="{next_x-30}" y2="150" stroke="black" stroke-width="3"/>'
    
    last_x = x_positions[-1]
    svg += f'''
        <line x1="{last_x+30}" y1="150" x2="620" y2="150" stroke="black" stroke-width="3"/>
        <line x1="620" y1="150" x2="620" y2="200" stroke="black" stroke-width="3"/>
        <line x1="620" y1="200" x2="80" y2="200" stroke="black" stroke-width="3"/>
        <line x1="80" y1="200" x2="80" y2="175" stroke="black" stroke-width="3"/>
        
        <!-- Connect first resistor -->
        <line x1="105" y1="150" x2="{x_positions[0]-30}" y2="150" stroke="black" stroke-width="3"/>
    </svg>'''
    
    return svg


def generate_parallel_resistor_diagram(components):
    resistors = [c for c in components if c['type'] == 'resistor']
    voltage_sources = [c for c in components if c['type'] == 'voltage_source']
    
    vin = voltage_sources[0] if voltage_sources else {'name': 'Vin', 'value': 12}
    
    svg = f'''<svg width="600" height="400" viewBox="0 0 600 400">
        <rect width="100%" height="100%" fill="#f8f9fa"/>
        <text x="300" y="25" text-anchor="middle" font-size="16" font-weight="bold">Parallel Resistor Circuit</text>
        
        <!-- Voltage Source -->
        <circle cx="80" cy="200" r="25" fill="white" stroke="black" stroke-width="2"/>
        <text x="80" y="195" text-anchor="middle" font-size="16" font-weight="bold">+</text>
        <text x="80" y="215" text-anchor="middle" font-size="16" font-weight="bold">-</text>
        <text x="80" y="245" text-anchor="middle" font-size="12" font-weight="bold">{vin['name']}</text>
        <text x="80" y="260" text-anchor="middle" font-size="10">{vin['value']}V</text>
        
        <!-- Bus lines -->
        <line x1="105" y1="150" x2="500" y2="150" stroke="black" stroke-width="4"/>
        <line x1="105" y1="250" x2="500" y2="250" stroke="black" stroke-width="4"/>
        
        <!-- Connect voltage source to buses -->
        <line x1="80" y1="175" x2="80" y2="150" stroke="black" stroke-width="3"/>
        <line x1="80" y1="150" x2="105" y2="150" stroke="black" stroke-width="3"/>
        <line x1="80" y1="225" x2="80" y2="250" stroke="black" stroke-width="3"/>
        <line x1="80" y1="250" x2="105" y2="250" stroke="black" stroke-width="3"/>
        
        <!-- Parallel resistors -->'''
    
    colors = ['#FF6B35', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    x_positions = [180 + i * 80 for i in range(min(len(resistors), 4))]
    
    for i, (resistor, x) in enumerate(zip(resistors[:4], x_positions)):
        svg += f'''
        <!-- Resistor {i+1} -->
        <line x1="{x}" y1="150" x2="{x}" y2="170" stroke="black" stroke-width="2"/>
        <rect x="{x-25}" y="170" width="50" height="20" fill="{colors[i]}" stroke="black" stroke-width="2" rx="3"/>
        <line x1="{x}" y1="190" x2="{x}" y2="250" stroke="black" stroke-width="2"/>
        <text x="{x}" y="140" text-anchor="middle" font-size="12" font-weight="bold">{resistor['name']}</text>
        <text x="{x}" y="270" text-anchor="middle" font-size="10">{format_value(resistor['value'], 'Ω')}</text>'''
    
    svg += '</svg>'
    return svg


def generate_voltage_distribution_plot(voltages):
    return "data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCI+PC9zdmc+"

def generate_power_distribution_plot(power_data):
    names = [item['name'] for item in power_data]
    powers = [item['power'] * 1000 for item in power_data]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, powers, color=plt.cm.Set3(range(len(names))))
    
    ax.set_ylabel('Power (mW)', fontsize=12)
    ax.set_title('Power Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{power:.2f}mW', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_data}"
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
    return f"{value:.3f}{unit}"

def parse_value(value_str, prefix, component_type=None):
    try:
        value = float(value_str)
        multipliers = {
            'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'μ': 1e-6,
            'm': 1e-3, 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12
        }
        
        if prefix and prefix in multipliers:
            value *= multipliers[prefix]
        
        return value
    except (ValueError, TypeError):
        return 1.0

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
    logger.info("Starting Advanced Circuit Analysis System...")
    logger.info(f"MATLAB Available: {MATLAB_AVAILABLE}")
    logger.info(f"MATLAB Engine Status: {'Connected' if matlab_mgr.is_connected() else 'Disconnected'}")
    port = int(os.environ.get('PORT', 5000))
    try:
        app.run(debug=True, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        matlab_mgr.disconnect()
        logger.info("Application stopped")
