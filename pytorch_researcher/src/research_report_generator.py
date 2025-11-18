#!/usr/bin/env python3
"""
Comprehensive Research Report Generator

This module generates professional, high-grade research reports in markdown format
from PyTorch ML Research Agent runs, including memory system insights and detailed analysis.

Features:
- Comprehensive research methodology documentation
- Detailed iteration analysis with technical insights
- Memory system integration reporting
- Performance metrics and evaluation results
- Failure analysis and pattern recognition
- Professional formatting with tables, charts, and structured data
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import memory system components for fetching actual memory content
try:
    from pytorch_researcher.src.memory import ManualMemoryContextManager
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


class ResearchReportGenerator:
    """Generate comprehensive research reports from agent run data."""
    
    def __init__(self, run_directory: str):
        """
        Initialize the report generator.
        
        Args:
            run_directory: Path to the research run directory
        """
        self.run_dir = Path(run_directory)
        self.registry_path = self.run_dir / "experiments" / "registry.json"
        self.report_path = self.run_dir / "RESEARCH_REPORT.md"
        self.memory_manager = None
        
        # Try to initialize memory manager for fetching memory content
        if MEMORY_AVAILABLE:
            try:
                # Create a basic disabled memory manager as fallback
                from memori import Memori
                disabled_memori = Memori(conscious_ingest=False, auto_ingest=False)
                self.memory_manager = ManualMemoryContextManager(disabled_memori)
            except Exception:
                self.memory_manager = None
        else:
            self.memory_manager = None
    
    def _fetch_memory_content(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the actual content of a memory by its ID.
        
        Args:
            memory_id: The memory ID to fetch
            
        Returns:
            Dictionary with memory content or None if not available
        """
        if not self.memory_manager:
            return None
            
        try:
            # Search for the memory using the ID
            search_results = self.memory_manager.search_memories(f"id:{memory_id}", limit=1)
            if search_results:
                return search_results[0]
        except Exception as e:
            print(f"Warning: Could not fetch memory content for ID {memory_id}: {e}")
        
        return None
    
    def _get_memory_content_by_ids(self, memory_ids: List[str]) -> Dict[str, Any]:
        """
        Get memory content for a list of memory IDs.
        
        Args:
            memory_ids: List of memory IDs to fetch
            
        Returns:
            Dictionary mapping memory ID to content
        """
        memory_contents = {}
        
        for memory_id in memory_ids:
            content = self._fetch_memory_content(memory_id)
            if content:
                memory_contents[memory_id] = content
            else:
                memory_contents[memory_id] = {"id": memory_id, "content": "Memory content not available"}
        
        return memory_contents
    
    def _format_memory_for_display(self, memory_data: Dict[str, Any], memory_id: str) -> str:
        """
        Format memory data for display in the report.
        
        Args:
            memory_data: Dictionary containing memory data
            memory_id: The memory ID for reference
            
        Returns:
            Formatted string for display
        """
        content = memory_data.get("content", "")
        if not content:
            content = memory_data.get("searchable_content", "")
        if not content:
            content = memory_data.get("summary", "No content available")
            
        category = memory_data.get("category_primary", "general")
        
        # Format the memory with ID reference and category
        formatted = f"**[{category.upper()}]** Memory ID: `{memory_id[:12]}...`\n"
        formatted += f"{content}\n"
        
        return formatted
        
    def generate_comprehensive_report(self, memory_insights: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a comprehensive research report.
        
        Args:
            memory_insights: Optional memory system insights to include
            
        Returns:
            Path to the generated report file
        """
        # Load research data
        registry_data = self._load_registry_data()
        research_data = self._parse_research_data(registry_data)
        
        # Generate report sections
        report_content = self._generate_report_header(research_data)
        report_content += self._generate_executive_summary(research_data)
        report_content += self._generate_research_methodology(research_data)
        report_content += self._generate_detailed_iterations(research_data)
        report_content += self._generate_performance_analysis(research_data)
        report_content += self._generate_memory_system_analysis(research_data, memory_insights)
        report_content += self._generate_failure_analysis(research_data)
        report_content += self._generate_recommendations(research_data)
        report_content += self._generate_appendices(research_data)
        
        # Write report to file
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        return str(self.report_path)
    
    def _load_registry_data(self) -> List[Dict[str, Any]]:
        """Load the registry JSON data."""
        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Registry file not found: {self.registry_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in registry file: {e}")
    
    def _parse_research_data(self, registry_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse and structure the research data."""
        research_data = {
            'goal': None,
            'iterations': [],
            'final_status': None,
            'total_iterations': 0,
            'successful_iterations': 0,
            'failed_iterations': 0,
            'planning_proposal': {},
            'memory_insights': {},
            'run_metadata': {}
        }
        
        for entry in registry_data:
            if 'goal' in entry:
                # This is the final run record
                research_data['goal'] = entry['goal']
                research_data['final_status'] = entry.get('report', {}).get('final_status', 'unknown')
                research_data['memory_insights'] = entry.get('memory_insights', {})
                research_data['run_metadata'] = {
                    'run_id': entry.get('run_id', ''),
                    'timestamp': entry.get('timestamp', ''),
                    'memory_enabled': entry.get('memory_enabled', False)
                }
                
                # Extract detailed data from the report
                report = entry.get('report', {})
                research_data['planning_proposal'] = report.get('planning_proposal', {})
                
                # Process iterations
                iterations = report.get('iterations', [])
                research_data['iterations'] = iterations
                research_data['total_iterations'] = len(iterations)
                
                for iteration in iterations:
                    if iteration.get('evaluation', {}).get('goal_achieved', False):
                        research_data['successful_iterations'] += 1
                    else:
                        research_data['failed_iterations'] += 1
                        
            elif 'iteration' in entry:
                # This is an iteration record
                research_data['iterations'].append(entry)
        
        return research_data
    
    def _generate_report_header(self, research_data: Dict[str, Any]) -> str:
        """Generate the report header section."""
        timestamp = research_data.get('run_metadata', {}).get('timestamp', '')
        run_id = research_data.get('run_metadata', {}).get('run_id', '')
        
        return f"""# ML Research Agent - Comprehensive Research Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Run ID**: {run_id}  
**Run Timestamp**: {timestamp}  
**Memory System**: {'✅ Enabled' if research_data.get('run_metadata', {}).get('memory_enabled') else '❌ Disabled'}  

---

## Research Objective

**Primary Goal**: {research_data.get('goal', 'No goal specified')}

This research was conducted using the PyTorch ML Research Agent, an autonomous machine learning research system capable of iterative experimentation, model generation, and performance optimization.

---

"""
    
    def _generate_executive_summary(self, research_data: Dict[str, Any]) -> str:
        """Generate the executive summary section."""
        total_iters = research_data.get('total_iterations', 0)
        successful_iters = research_data.get('successful_iterations', 0)
        failed_iters = research_data.get('failed_iterations', 0)
        final_status = research_data.get('final_status', 'unknown')
        
        success_rate = (successful_iters/total_iters*100 if total_iters > 0 else 0)
        
        return f"""## Executive Summary

### Research Overview
- **Total Iterations**: {total_iters}
- **Successful Iterations**: {successful_iters}
- **Failed Iterations**: {failed_iters}
- **Success Rate**: {success_rate:.1f}%
- **Final Status**: {final_status.replace('_', ' ').title()}

### Key Findings

"""
    
    def _generate_research_methodology(self, research_data: Dict[str, Any]) -> str:
        """Generate the research methodology section."""
        proposal = research_data.get('planning_proposal', {})
        model_config = proposal.get('model_config', {})
        evaluation_config = proposal.get('evaluation_config', {})
        
        methodology = f"""## Research Methodology

### Planning Phase
The research began with the Planning LLM generating an initial configuration proposal based on the research goal and domain knowledge.

#### Proposed Model Architecture
- **Architecture Type**: {model_config.get('architecture', 'Unknown')}
- **Framework**: {model_config.get('framework', 'PyTorch')}
- **Layers**: {len(model_config.get('layers', []))} defined layers

#### Model Configuration Details

| Component | Specification |
|-----------|---------------|
| Architecture | {model_config.get('architecture', 'N/A')} |
| Optimizer | {model_config.get('optimizer', 'N/A')} |
| Loss Function | {model_config.get('loss', 'N/A')} |
| Total Layers | {len(model_config.get('layers', []))} |

#### Layer Architecture

"""
        
        # Add layer details if available
        layers = model_config.get('layers', [])
        if layers:
            methodology += "| Layer Type | Configuration | Details |\n"
            methodology += "|------------|---------------|----------|\n"
            
            for i, layer in enumerate(layers, 1):
                layer_type = layer.get('type', 'Unknown')
                details = []
                if 'filters' in layer:
                    details.append(f"Filters: {layer['filters']}")
                if 'kernel_size' in layer:
                    details.append(f"Kernel: {layer['kernel_size']}")
                if 'activation' in layer:
                    details.append(f"Activation: {layer['activation']}")
                if 'units' in layer:
                    details.append(f"Units: {layer['units']}")
                    
                methodology += f"| {i}. {layer_type} | {', '.join(details) if details else 'Basic'} | {layer} |\n"
        
        methodology += f"""

#### Evaluation Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | {evaluation_config.get('dataset_name', 'N/A')} |
| Subset Size | {evaluation_config.get('subset_size', 'N/A')} |
| Epochs | {evaluation_config.get('epochs', 'N/A')} |
| Batch Size | {evaluation_config.get('batch_size', 'N/A')} |
| Target Accuracy | {evaluation_config.get('target_accuracy', 'N/A')} |

### Iterative Research Process

The research followed an iterative methodology with the following phases:

1. **Model Assembly**: LLM-generated model code based on configuration
2. **Sandbox Validation**: Automated testing of model architecture
3. **Performance Evaluation**: Training and validation on specified dataset
4. **Decision Making**: Planning LLM analysis of results and next steps
5. **Iteration Continuation**: Refinement based on performance and insights

---

"""
        
        return methodology
    
    def _generate_detailed_iterations(self, research_data: Dict[str, Any]) -> str:
        """Generate detailed iteration analysis."""
        iterations = research_data.get('iterations', [])
        
        if not iterations:
            return "## Detailed Iterations\n\nNo iterations recorded.\n\n"
        
        content = "## Detailed Iteration Analysis\n\n"
        
        for i, iteration in enumerate(iterations, 1):
            iteration_num = iteration.get('iteration', i)
            model_config = iteration.get('model_config', {})
            evaluation = iteration.get('evaluation', {})
            sandbox = iteration.get('sandbox', {})
            assemble = iteration.get('assemble', {})
            
            content += f"""### Iteration {iteration_num}

#### Model Configuration
```
{json.dumps(model_config, indent=2)}
```

#### Model Assembly
- **Method**: {assemble.get('via', 'Unknown')}
- **Source**: {assemble.get('source', 'No source code')[:200]}...
- **Assembly Path**: {assemble.get('path', 'N/A')}
- **Attempts**: {assemble.get('attempts', 'N/A')}

#### Sandbox Validation
- **Status**: {'✅ PASSED' if sandbox.get('success') else '❌ FAILED'}
- **Return Code**: {sandbox.get('returncode', 'N/A')}
- **Duration**: {sandbox.get('duration', 'N/A') if isinstance(sandbox.get('duration'), (int, float)) else sandbox.get('duration', 'N/A')}{'' if sandbox.get('duration') == 'N/A' else 's'}
"""

            if sandbox.get('success'):
                parsed = sandbox.get('parsed', {})
                total_params = parsed.get('total_params', 'N/A')
                if isinstance(total_params, (int, float)):
                    total_params_str = f"{total_params:,}"
                else:
                    total_params_str = str(total_params)
                content += f"""- **Model Class**: {parsed.get('class_name', 'N/A')}
- **Total Parameters**: {total_params_str}
- **Output Shape**: {parsed.get('output_shape', 'N/A')}
"""
            else:
                error = sandbox.get('stdout', '')
                if error:
                    content += f"- **Error**: ```{error[:500]}...```\n"

            content += f"""
#### Performance Evaluation
"""
            
            if evaluation and not evaluation.get('evaluation_error'):
                final_metrics = evaluation.get('final', {})
                aggregated = evaluation.get('aggregated', {})
                
                # Safely format numeric values
                val_accuracy = final_metrics.get('val_accuracy', 'N/A')
                val_loss = final_metrics.get('val_loss', 'N/A')
                
                if isinstance(val_accuracy, (int, float)):
                    val_accuracy_str = f"{val_accuracy:.4f}"
                else:
                    val_accuracy_str = str(val_accuracy)
                    
                if isinstance(val_loss, (int, float)):
                    val_loss_str = f"{val_loss:.4f}"
                else:
                    val_loss_str = str(val_loss)
                
                content += f"""- **Final Validation Accuracy**: {val_accuracy_str}
- **Final Validation Loss**: {val_loss_str}
- **Goal Achieved**: {'✅ Yes' if evaluation.get('goal_achieved') else '❌ No'}

##### Detailed Metrics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
"""

                # Safely format aggregated metrics
                def format_metric(metric_data, key):
                    value = metric_data.get(key, 'N/A')
                    if isinstance(value, (int, float)):
                        return f"{value:.4f}"
                    return str(value)
                
                val_acc_stats = aggregated.get('val_accuracy', {})
                val_loss_stats = aggregated.get('val_loss', {})
                
                content += f"| Val Accuracy | {format_metric(val_acc_stats, 'mean')} | {format_metric(val_acc_stats, 'std')} | {format_metric(val_acc_stats, 'min')} | {format_metric(val_acc_stats, 'max')} |\n"
                content += f"| Val Loss | {format_metric(val_loss_stats, 'mean')} | {format_metric(val_loss_stats, 'std')} | {format_metric(val_loss_stats, 'min')} | {format_metric(val_loss_stats, 'max')} |\n"
                
                # Add training history if available
                seed_results = evaluation.get('seed_results', [])
                if seed_results and seed_results[0].get('history'):
                    history = seed_results[0]['history']
                    content += "##### Training History\n\n"
                    content += "| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |\n"
                    content += "|-------|------------|-----------|----------|---------|\n"
                    
                    for epoch_data in history:
                        # Safely format epoch data
                        epoch = epoch_data.get('epoch', 'N/A')
                        train_loss = epoch_data.get('train', {}).get('loss', 'N/A')
                        train_acc = epoch_data.get('train', {}).get('accuracy', 'N/A')
                        val_loss = epoch_data.get('val', {}).get('val_loss', 'N/A')
                        val_acc = epoch_data.get('val', {}).get('val_accuracy', 'N/A')
                        
                        # Format numeric values safely
                        if isinstance(train_loss, (int, float)):
                            train_loss_str = f"{train_loss:.4f}"
                        else:
                            train_loss_str = str(train_loss)
                            
                        if isinstance(train_acc, (int, float)):
                            train_acc_str = f"{train_acc:.4f}"
                        else:
                            train_acc_str = str(train_acc)
                            
                        if isinstance(val_loss, (int, float)):
                            val_loss_str = f"{val_loss:.4f}"
                        else:
                            val_loss_str = str(val_loss)
                            
                        if isinstance(val_acc, (int, float)):
                            val_acc_str = f"{val_acc:.4f}"
                        else:
                            val_acc_str = str(val_acc)
                        
                        content += f"| {epoch} | {train_loss_str} | {train_acc_str} | {val_loss_str} | {val_acc_str} |\n"
            else:
                content += f"- **Evaluation Error**: {evaluation.get('evaluation_error', 'Unknown error')}\n"

            content += "\n---\n\n"
        
        return content
    
    def _generate_performance_analysis(self, research_data: Dict[str, Any]) -> str:
        """Generate performance analysis section."""
        iterations = research_data.get('iterations', [])
        
        if not iterations:
            return "## Performance Analysis\n\nNo iterations to analyze.\n\n"
        
        content = "## Performance Analysis\n\n"
        
        # Collect performance data
        performance_data = []
        for iteration in iterations:
            iteration_num = iteration.get('iteration', 0)
            evaluation = iteration.get('evaluation', {})
            if evaluation and not evaluation.get('evaluation_error'):
                final_metrics = evaluation.get('final', {})
                performance_data.append({
                    'iteration': iteration_num,
                    'accuracy': final_metrics.get('val_accuracy'),
                    'loss': final_metrics.get('val_loss'),
                    'goal_achieved': evaluation.get('goal_achieved', False)
                })
        
        if performance_data:
            content += "### Accuracy Progression\n\n"
            content += "| Iteration | Validation Accuracy | Validation Loss | Goal Achieved |\n"
            content += "|-----------|-------------------|----------------|---------------|\n"
            
            for data in performance_data:
                achieved = "✅" if data['goal_achieved'] else "❌"
                
                # Safely format accuracy and loss
                accuracy = data['accuracy']
                loss = data['loss']
                
                if isinstance(accuracy, (int, float)):
                    accuracy_str = f"{accuracy:.4f}"
                else:
                    accuracy_str = str(accuracy)
                    
                if isinstance(loss, (int, float)):
                    loss_str = f"{loss:.4f}"
                else:
                    loss_str = str(loss)
                
                content += f"| {data['iteration']} | {accuracy_str} | {loss_str} | {achieved} |\n"
            
            # Calculate statistics
            accuracies = [d['accuracy'] for d in performance_data if d['accuracy'] is not None and isinstance(d['accuracy'], (int, float))]
            if accuracies:
                best_acc = max(accuracies)
                worst_acc = min(accuracies)
                mean_acc = sum(accuracies)/len(accuracies)
                improvement = ((best_acc - worst_acc)/worst_acc*100 if worst_acc > 0 else 0)
                
                content += f"""### Statistical Summary

- **Best Accuracy**: {best_acc:.4f}
- **Worst Accuracy**: {worst_acc:.4f}
- **Mean Accuracy**: {mean_acc:.4f}
- **Improvement**: {improvement:.1f}%
"""

        content += f"""
### Iteration Outcomes

- **Total Iterations**: {len(iterations)}
- **Successful Evaluations**: {len(performance_data)}
- **Failed Evaluations**: {len(iterations) - len(performance_data)}
- **Goals Achieved**: {sum(1 for d in performance_data if d['goal_achieved'])}

---

"""
        
        return content
    
    def _generate_memory_system_analysis(self, research_data: Dict[str, Any], memory_insights: Optional[Dict[str, Any]]) -> str:
        """Generate memory system analysis section."""
        content = "## Research Insights & Analysis\n\n"
        
        memory_enabled = research_data.get('run_metadata', {}).get('memory_enabled', False)
        
        if not memory_enabled:
            content += "❌ **Memory System Disabled**: No research insights were recorded for this run.\n\n"
            return content
        
        content += "🧠 **Memory System Active**: This research utilized conscious memory management for enhanced decision-making.\n\n"
        
        # Memory insights - show actual content, not just IDs
        insights = research_data.get('memory_insights', {})
        if memory_insights:
            insights.update(memory_insights)
        
        if insights:
            content += "### Memory Insights Recorded\n\n"
            content += "The following research insights were captured and utilized during the research process:\n\n"
            
            # Group insights by type and fetch actual content
            for insight_type, insight_data in insights.items():
                insight_name = insight_type.replace('_', ' ').title()
                content += f"#### {insight_name}\n\n"
                
                # Handle different types of insight data
                if isinstance(insight_data, str) and len(insight_data) == 36:  # Looks like a UUID
                    # This is a memory ID, fetch the actual content
                    memory_content = self._fetch_memory_content(insight_data)
                    if memory_content:
                        formatted_content = self._format_memory_for_display(memory_content, insight_data)
                        content += formatted_content
                    else:
                        content += f"**Memory ID**: `{insight_data}` (content not available)\n\n"
                elif isinstance(insight_data, dict):
                    # It's a dictionary with memory data
                    memory_id = insight_data.get('id', 'unknown')
                    formatted_content = self._format_memory_for_display(insight_data, memory_id)
                    content += formatted_content
                else:
                    # Display the raw data
                    content += f"**Raw Data**: {str(insight_data)}\n\n"
        
        # Add section about memory usage by research phase
        content += "### Memory Usage by Research Phase\n\n"
        
        # Check if we have memory usage tracking data
        memory_usage_by_phase = research_data.get('memory_usage_by_phase', {})
        
        if memory_usage_by_phase:
            content += "The following memory resources were actively utilized during research:\n\n"
            
            for phase, memory_ids in memory_usage_by_phase.items():
                phase_name = phase.replace('_', ' ').title()
                content += f"#### {phase_name} Phase\n\n"
                content += f"**Memory Resources Used**: {len(memory_ids)}\n"
                
                # List memory IDs used in this phase
                for memory_id in memory_ids:
                    content += f"- Memory ID: `{memory_id[:12]}...`\n"
                content += "\n"
        else:
            content += "Memory system was available for consultation during research phases, "
            content += "but detailed usage tracking was not recorded.\n\n"
        
        # Enhanced memory insights with actual content
        content += "### Detailed Memory Insights\n\n"
        
        if insights:
            for insight_type, insight_data in insights.items():
                insight_name = insight_type.replace('_', ' ').title()
                content += f"#### {insight_name}\n\n"
                
                # Handle different types of insight data
                if isinstance(insight_data, str) and len(insight_data) == 36:  # Looks like a UUID
                    # This is a memory ID, fetch the actual content
                    memory_content = self._fetch_memory_content(insight_data)
                    if memory_content:
                        formatted_content = self._format_memory_for_display(memory_content, insight_data)
                        content += formatted_content
                    else:
                        content += f"**Memory ID**: `{insight_data}` (content not available)\n\n"
                elif isinstance(insight_data, dict):
                    # It's a dictionary with memory data
                    memory_id = insight_data.get('id', 'unknown')
                    formatted_content = self._format_memory_for_display(insight_data, memory_id)
                    content += formatted_content
                else:
                    # Display the raw data
                    content += f"**Raw Data**: {str(insight_data)}\n\n"
        
        # Add information about the memory management approach
        content += "### Memory Management Approach\n\n"
        content += "This research used **conscious memory management** with the following characteristics:\n\n"
        content += "- **Conscious Ingestion**: Enabled for strategic memory processing\n"
        content += "- **No Auto-Interception**: Memory operations remain explicit and controlled\n"
        content += "- **Strategic Context Injection**: Memory context injected at key decision points\n"
        content += "- **Cross-Session Persistence**: Memory insights tracked across research sessions\n"
        content += "- **Phase-Aware Usage**: Memory context tailored to specific research phases\n\n"

        return content
    
    def _generate_failure_analysis(self, research_data: Dict[str, Any]) -> str:
        """Generate failure analysis section."""
        iterations = research_data.get('iterations', [])
        
        content = "## Failure Analysis\n\n"
        
        failed_iterations = []
        for iteration in iterations:
            iteration_num = iteration.get('iteration', 0)
            sandbox = iteration.get('sandbox', {})
            evaluation = iteration.get('evaluation', {})
            
            failure_reasons = []
            
            if not sandbox.get('success'):
                failure_reasons.append("Sandbox validation failed")
            
            if evaluation and evaluation.get('evaluation_error'):
                failure_reasons.append("Evaluation error")
            
            if failure_reasons:
                failed_iterations.append({
                    'iteration': iteration_num,
                    'reasons': failure_reasons,
                    'sandbox_error': sandbox.get('stdout', ''),
                    'evaluation_error': evaluation.get('evaluation_error', '')
                })
        
        if not failed_iterations:
            content += "✅ **No failures recorded during this research run.**\n\n"
            return content
        
        content += f"**Total Failed Iterations**: {len(failed_iterations)}\n\n"
        
        for failure in failed_iterations:
            content += f"""### Iteration {failure['iteration']} Failures

**Failure Reasons**:
"""
            for reason in failure['reasons']:
                content += f"- {reason}\n"
            
            if failure['sandbox_error']:
                content += f"\n**Sandbox Error**:\n```\n{failure['sandbox_error'][:500]}...\n```\n"
            
            if failure['evaluation_error']:
                content += f"\n**Evaluation Error**: {failure['evaluation_error']}\n"
            
            content += "\n---\n\n"
        
        return content
    
    def _generate_recommendations(self, research_data: Dict[str, Any]) -> str:
        """Generate recommendations section."""
        content = "## Recommendations\n\n"
        
        final_status = research_data.get('final_status', '')
        total_iters = research_data.get('total_iterations', 0)
        successful_iters = research_data.get('successful_iterations', 0)
        
        if final_status == 'achieved':
            content += "### ✅ Research Goal Achieved\n\n"
            content += "The research successfully achieved its objective. Consider:\n\n"
            content += "1. **Documentation**: Document the successful approach for future reference\n"
            content += "2. **Generalization**: Test the approach on similar problems\n"
            content += "3. **Optimization**: Fine-tune hyperparameters for even better performance\n"
            content += "4. **Memory Integration**: Leverage the recorded insights for related research\n"
            
        elif final_status == 'max_iterations_reached':
            content += "### ⏰ Maximum Iterations Reached\n\n"
            content += "The research reached the iteration limit without achieving the goal. Recommendations:\n\n"
            content += "1. **Extend Iterations**: Increase `max_iter` parameter for more attempts\n"
            content += "2. **Architecture Review**: Examine the model architecture for fundamental issues\n"
            content += "3. **Dataset Analysis**: Verify dataset compatibility and quality\n"
            content += "4. **Memory Insights**: Review recorded failure patterns for systematic issues\n"
            content += "5. **Hyperparameter Tuning**: Adjust learning rate, batch size, or other parameters\n"
            
        else:
            content += f"### 📊 Research Status: {final_status.replace('_', ' ').title()}\n\n"
            content += "Recommendations based on research outcomes:\n\n"
            
            if successful_iters > 0:
                content += f"1. **Build on Success**: {successful_iters} iteration(s) showed promise\n"
                content += "2. **Refine Approach**: Focus on the aspects that worked in successful iterations\n"
                content += "3. **Memory Leverage**: Use recorded insights to guide refinements\n"
            else:
                content += "1. **Fundamental Review**: Consider reviewing the research approach\n"
                content += "2. **Domain Knowledge**: Consult domain experts for alternative approaches\n"
                content += "3. **Resource Check**: Verify computational resources and constraints\n"
        
        content += """
### Future Research Directions

Based on the memory system insights and research outcomes:

1. **Leverage Memory Insights**: Use recorded research patterns for similar problems
2. **Cross-Domain Application**: Apply successful approaches to related domains
3. **Automated Refinement**: Develop automated hyperparameter tuning based on patterns
4. **Knowledge Transfer**: Use insights from this research to bootstrap future projects
5. **Memory-Enhanced Planning**: Continue using memory system for strategic research decisions

---

"""
        
        return content
    
    def _generate_appendices(self, research_data: Dict[str, Any]) -> str:
        """Generate appendices section."""
        content = "## Appendices\n\n"
        
        # Technical details
        content += "### A. Technical Implementation Details\n\n"
        content += f"- **Agent Version**: PyTorch ML Research Agent\n"
        content += f"- **Research Iterations**: {research_data.get('total_iterations', 0)}\n"
        content += f"- **Memory System**: {'Enabled' if research_data.get('run_metadata', {}).get('memory_enabled') else 'Disabled'}\n"
        content += f"- **Planning LLM**: OpenRouter (Sherlock-Dash-Alpha)\n"
        content += f"- **Framework**: PyTorch\n"
        content += f"- **Evaluation**: QuickEval with configurable parameters\n\n"
        
        # Configuration details
        proposal = research_data.get('planning_proposal', {})
        content += "### B. Planning LLM Proposal\n\n"
        content += f"```json\n{json.dumps(proposal, indent=2)}\n```\n\n"
        
        # Raw iteration data
        iterations = research_data.get('iterations', [])
        if iterations:
            content += "### C. Complete Iteration Data\n\n"
            for i, iteration in enumerate(iterations, 1):
                content += f"#### Iteration {i} Complete Data\n\n"
                content += f"```json\n{json.dumps(iteration, indent=2)}\n```\n\n"
        
        content += f"""### D. Report Generation

- **Generated by**: Research Report Generator v1.0
- **Generation Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
- **Source Data**: {self.registry_path}
- **Report Location**: {self.report_path}

---

## End of Report

*This report was automatically generated by the PyTorch ML Research Agent research report generator.*
"""
        
        return content


def generate_research_report(run_directory: str, memory_insights: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a comprehensive research report for a given run directory.
    
    Args:
        run_directory: Path to the research run directory
        memory_insights: Optional memory system insights to include
        
    Returns:
        Path to the generated report file
    """
    generator = ResearchReportGenerator(run_directory)
    return generator.generate_comprehensive_report(memory_insights)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python research_report_generator.py <run_directory>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    memory_insights = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    
    try:
        report_path = generate_research_report(run_dir, memory_insights)
        print(f"✅ Research report generated: {report_path}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        sys.exit(1)