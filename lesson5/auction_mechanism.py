#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auction Mechanism implementation for the Task Allocation and Auction Mechanisms lesson.

This module implements various auction mechanisms for multi-robot task allocation,
including single-item auctions, combinatorial auctions, and different payment rules.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple, Union, Optional, Any, Callable
import random
import time
import copy
from collections import defaultdict
import networkx as nx
from dataclasses import dataclass, field
import sys


class AuctionMechanism:
    """
    A class implementing various auction mechanisms for task allocation.
    
    This class supports different auction types (sequential, parallel, combinatorial) and
    payment rules (first-price, second-price, VCG), enabling the exploration of different
    market-based task allocation approaches.
    """
    
    def __init__(self, 
                 auction_type: str = 'sequential',
                 payment_rule: str = 'first_price',
                 max_bundle_size: int = 3,
                 reserve_price: float = 0.0,
                 timeout: float = 30.0,
                 communication_cost: float = 0.0):
        """
        Initialize the auction mechanism with specified parameters.
        
        Args:
            auction_type: Type of auction ('sequential', 'parallel', 'combinatorial')
            payment_rule: Rule for determining payments ('first_price', 'second_price', 'vcg')
            max_bundle_size: Maximum number of tasks in a bundle for combinatorial auctions
            reserve_price: Minimum price for tasks to be allocated
            timeout: Maximum time (seconds) to run winner determination
            communication_cost: Cost per message sent during the auction
        """
        self.auction_type = auction_type
        self.payment_rule = payment_rule
        self.max_bundle_size = max_bundle_size
        self.reserve_price = reserve_price
        self.timeout = timeout
        self.communication_cost = communication_cost
        
        # Auction history
        self.auction_history = []
        
        # Performance metrics
        self.metrics = {
            'social_welfare': [],
            'revenue': [],
            'allocation_efficiency': [],
            'execution_time': [],
            'message_count': []
        }
        
        # Verify valid combinations
        if auction_type == 'combinatorial' and payment_rule not in ['vcg', 'first_price']:
            print(f"Warning: {payment_rule} payment rule may not be incentive-compatible with combinatorial auctions")
    
    def announce_tasks(self, available_tasks: List) -> Dict:
        """
        Announce tasks that are available for bidding.
        
        Args:
            available_tasks: List of Task objects available for allocation
            
        Returns:
            Dict: Announcement structure with task details
        """
        # Create task announcement dictionary
        announcement = {
            'auction_id': f"auction_{int(time.time())}_{random.randint(0, 1000)}",
            'auction_type': self.auction_type,
            'payment_rule': self.payment_rule,
            'tasks': [],
            'timestamp': time.time()
        }
        
        # Structure tasks based on auction type
        if self.auction_type == 'sequential':
            # For sequential auctions, announce one task at a time
            announcement['tasks'] = [self._create_task_announcement(task) for task in available_tasks]
            announcement['sequential'] = True
            
        elif self.auction_type == 'parallel':
            # For parallel auctions, announce all tasks at once, but each is bid on separately
            announcement['tasks'] = [self._create_task_announcement(task) for task in available_tasks]
            announcement['sequential'] = False
            
        elif self.auction_type == 'combinatorial':
            # For combinatorial auctions, group tasks into bundles
            announcement['tasks'] = [self._create_task_announcement(task) for task in available_tasks]
            
            # Create potential bundles for bidding
            bundles = self._create_bundles(available_tasks)
            announcement['bundles'] = bundles
            announcement['sequential'] = False
            
        # Track message count
        self.metrics['message_count'].append(1)  # One announcement message
        
        return announcement
    
    def _create_task_announcement(self, task) -> Dict:
        """Create a structured announcement for a single task."""
        # Basic task information
        task_info = {
            'id': task.id,
            'type': task.type,
            'position': task.position,
            'priority': task.priority,
            'required_capabilities': task.required_capabilities,
            'completion_time': task.completion_time,
            'reserve_price': self.reserve_price
        }
        
        # Add optional information if available
        if hasattr(task, 'deadline') and task.deadline is not None:
            task_info['deadline'] = task.deadline
            
        if hasattr(task, 'requires_coalition') and task.requires_coalition:
            task_info['requires_coalition'] = True
            task_info['role_requirements'] = task.role_requirements
            
        if hasattr(task, 'bundle_id') and task.bundle_id:
            task_info['bundle_id'] = task.bundle_id
            
        return task_info
    
    def _create_bundles(self, tasks: List) -> List[Dict]:
        """Create potential bundles for combinatorial auctions."""
        bundles = []
        
        # First, add all predefined bundles
        predefined_bundles = defaultdict(list)
        for task in tasks:
            if hasattr(task, 'bundle_id') and task.bundle_id:
                predefined_bundles[task.bundle_id].append(task.id)
                
        for bundle_id, task_ids in predefined_bundles.items():
            bundles.append({
                'id': bundle_id,
                'task_ids': task_ids,
                'size': len(task_ids)
            })
        
        # Then, create bundles based on spatial proximity
        unbundled_tasks = [task for task in tasks if not (hasattr(task, 'bundle_id') and task.bundle_id)]
        
        # Create a graph where nodes are tasks and edges connect nearby tasks
        G = nx.Graph()
        for task in unbundled_tasks:
            G.add_node(task.id, position=task.position)
            
        # Add edges between nearby tasks
        for i, task1 in enumerate(unbundled_tasks):
            for j, task2 in enumerate(unbundled_tasks[i+1:], i+1):
                # Calculate distance between tasks
                dist = ((task1.position[0] - task2.position[0])**2 + 
                        (task1.position[1] - task2.position[1])**2)**0.5
                
                # Add edge if tasks are close
                if dist < 10:  # Threshold for proximity
                    G.add_edge(task1.id, task2.id, weight=dist)
        
        # Find cliques (fully connected subgraphs) to create bundles
        cliques = list(nx.find_cliques(G))
        for i, clique in enumerate(cliques):
            if 2 <= len(clique) <= self.max_bundle_size:  # Only bundles of reasonable size
                bundles.append({
                    'id': f"proximity_bundle_{i}",
                    'task_ids': clique,
                    'size': len(clique)
                })
                
        return bundles
    
    def collect_bids(self, robots: List, tasks: List, 
                     announcement: Dict = None) -> Dict[str, Dict]:
        """
        Collect bids from robots for the announced tasks.
        
        Args:
            robots: List of robot objects that can bid
            tasks: List of tasks being auctioned
            announcement: The task announcement (if None, a new one is created)
            
        Returns:
            Dict: Collected bids from all robots
        """
        if announcement is None:
            announcement = self.announce_tasks(tasks)
            
        all_bids = {}
        message_count = 0
        
        # Map tasks by id for easy lookup
        task_map = {task.id: task for task in tasks}
        
        # Process based on auction type
        if self.auction_type in ['sequential', 'parallel']:
            # Collect bids for individual tasks
            for task_info in announcement['tasks']:
                task_id = task_info['id']
                task = task_map.get(task_id)
                
                if task is None:
                    continue
                
                # Collect bids from all robots
                task_bids = []
                for robot in robots:
                    # Generate bid if the robot is capable
                    print(f"Robot {robot.id} ({robot.strategy_type}) evaluating task {task_id} (type: {task.type})")
                    bid = robot.generate_bid(task)
                    print(f"Robot {robot.id} bid {bid} for task {task_id}")
                    if bid is not None and bid > self.reserve_price:
                        task_bids.append({
                            'robot_id': robot.id,
                            'task_id': task_id,
                            'bid_value': bid
                        })
                        message_count += 1  # Count each bid as a message
                
                all_bids[task_id] = task_bids
                
                # In sequential auctions, determine winner after each task
                if self.auction_type == 'sequential':
                    # If this is a sequential auction, we would determine winner here
                    # but for this implementation we'll do it later for all tasks
                    pass
        
        elif self.auction_type == 'combinatorial':
            # Collect bids for individual tasks and bundles
            single_bids = {}
            bundle_bids = []
            
            for robot in robots:
                robot_single_bids = {}
                robot_bundle_bids = []
                
                # Bid on individual tasks
                for task_info in announcement['tasks']:
                    task_id = task_info['id']
                    task = task_map.get(task_id)
                    
                    if task is None:
                        continue
                    
                    # Generate single-item bid
                    bid_result = robot.generate_bid(task, auction_type='combinatorial')
                    
                    if isinstance(bid_result, dict) and 'single' in bid_result:
                        if bid_result['single'] > self.reserve_price:
                            robot_single_bids[task_id] = bid_result['single']
                            message_count += 1
                    elif bid_result is not None and bid_result > self.reserve_price:
                        robot_single_bids[task_id] = bid_result
                        message_count += 1
                
                # Bid on bundles
                for bundle in announcement.get('bundles', []):
                    bundle_task_ids = bundle['task_ids']
                    bundle_tasks = [task_map.get(tid) for tid in bundle_task_ids if tid in task_map]
                    
                    if not bundle_tasks:
                        continue
                    
                    # Calculate bundle value based on synergy
                    bundle_value = 0
                    for task in bundle_tasks:
                        # Get individual value, either from earlier bid or generate new
                        if task.id in robot_single_bids:
                            task_value = robot_single_bids[task.id]
                        else:
                            bid_result = robot.generate_bid(task)
                            task_value = bid_result if bid_result is not None else 0
                        
                        bundle_value += task_value
                    
                    # Add synergy bonus (robots might value bundles more than sum of parts)
                    # This would ideally come from the robot's evaluation, but we synthesize it here
                    synergy_factor = 1.2  # Bundle worth 20% more than sum of parts
                    bundle_value *= synergy_factor
                    
                    if bundle_value > self.reserve_price * len(bundle_task_ids):
                        robot_bundle_bids.append({
                            'robot_id': robot.id,
                            'bundle_id': bundle['id'],
                            'task_ids': bundle_task_ids,
                            'bid_value': bundle_value
                        })
                        message_count += 1
                
                # Add robot's bids to overall collection
                for task_id, bid_value in robot_single_bids.items():
                    if task_id not in single_bids:
                        single_bids[task_id] = []
                    single_bids[task_id].append({
                        'robot_id': robot.id,
                        'task_id': task_id,
                        'bid_value': bid_value
                    })
                
                bundle_bids.extend(robot_bundle_bids)
            
            all_bids = {
                'single_bids': single_bids,
                'bundle_bids': bundle_bids
            }
        
        # Update message count metric
        self.metrics['message_count'][-1] += message_count
        
        return all_bids
    
    def determine_winners(self, bids: Dict, tasks: List) -> Dict:
        """
        Determine winners of the auction based on bids.
        
        Args:
            bids: Dictionary of bids from robots
            tasks: List of tasks being auctioned
            
        Returns:
            Dict: Task allocation and payment information
        """
        start_time = time.time()
        
        # Results will contain allocation and payment information
        results = {
            'allocations': {},  # task_id -> robot_id
            'payments': {},     # task_id -> payment amount
            'unallocated': [],  # list of unallocated task ids
            'welfare': 0.0      # total social welfare
        }
        
        if self.auction_type in ['sequential', 'parallel']:
            # Process each task individually
            for task in tasks:
                task_id = task.id
                
                # Get bids for this task
                task_bids = bids.get(task_id, [])
                
                if not task_bids:
                    results['unallocated'].append(task_id)
                    continue
                
                # Sort bids by value (highest first)
                sorted_bids = sorted(task_bids, key=lambda b: b['bid_value'], reverse=True)
                
                # Winner is highest bidder
                winner = sorted_bids[0]
                winner_id = winner['robot_id']
                winner_bid = winner['bid_value']
                
                # Determine payment based on payment rule
                if self.payment_rule == 'first_price':
                    payment = winner_bid
                elif self.payment_rule == 'second_price':
                    # Second price (Vickrey) auction: pay second highest bid
                    if len(sorted_bids) > 1:
                        payment = sorted_bids[1]['bid_value']
                    else:
                        payment = self.reserve_price
                else:
                    # Default to first price
                    payment = winner_bid
                
                # Record allocation and payment
                results['allocations'][task_id] = winner_id
                results['payments'][task_id] = payment
                results['welfare'] += winner_bid  # Social welfare is value to winner
                
        elif self.auction_type == 'combinatorial':
            # Use combinatorial winner determination
            single_bids = bids.get('single_bids', {})
            bundle_bids = bids.get('bundle_bids', [])
            
            # Convert to flat list of bids (both single and bundle)
            all_bids = []
            
            # Add single-item bids
            for task_id, task_bids in single_bids.items():
                for bid in task_bids:
                    all_bids.append({
                        'type': 'single',
                        'robot_id': bid['robot_id'],
                        'task_ids': [bid['task_id']],
                        'bid_value': bid['bid_value']
                    })
            
            # Add bundle bids
            for bid in bundle_bids:
                all_bids.append({
                    'type': 'bundle',
                    'robot_id': bid['robot_id'],
                    'task_ids': bid['task_ids'],
                    'bid_value': bid['bid_value'],
                    'bundle_id': bid.get('bundle_id')
                })
            
            # Call appropriate winner determination algorithm
            if self.payment_rule == 'vcg':
                allocation, payments = self._vcg_winner_determination(all_bids, tasks)
            else:
                allocation = self._greedy_winner_determination(all_bids, tasks)
                payments = self._calculate_payments(allocation, all_bids)
            
            # Process allocation results
            for task_id, robot_id in allocation.items():
                results['allocations'][task_id] = robot_id
                
            # Add payment information
            for task_id, payment in payments.items():
                results['payments'][task_id] = payment
            
            # Identify unallocated tasks
            allocated_tasks = set(allocation.keys())
            results['unallocated'] = [task.id for task in tasks if task.id not in allocated_tasks]
            
            # Calculate social welfare (sum of winning bid values)
            winning_bids_value = 0
            for bid in all_bids:
                robot_id = bid['robot_id']
                all_allocated = True
                
                for task_id in bid['task_ids']:
                    if allocation.get(task_id) != robot_id:
                        all_allocated = False
                        break
                
                if all_allocated:
                    winning_bids_value += bid['bid_value']
            
            results['welfare'] = winning_bids_value
        
        # Calculate execution time
        execution_time = time.time() - start_time
        results['execution_time'] = execution_time
        self.metrics['execution_time'].append(execution_time)
        
        # Record metrics
        self.metrics['social_welfare'].append(results['welfare'])
        revenue = sum(results['payments'].values())
        self.metrics['revenue'].append(revenue)
        
        allocation_rate = (len(tasks) - len(results['unallocated'])) / len(tasks) if tasks else 0
        self.metrics['allocation_efficiency'].append(allocation_rate)
        
        # Add auction to history
        self.auction_history.append({
            'timestamp': time.time(),
            'auction_type': self.auction_type,
            'payment_rule': self.payment_rule,
            'task_count': len(tasks),
            'bid_count': sum(len(bids.get(t.id, [])) for t in tasks) if not self.auction_type == 'combinatorial' else len(all_bids),
            'allocated_count': len(tasks) - len(results['unallocated']),
            'welfare': results['welfare'],
            'revenue': revenue,
            'execution_time': execution_time
        })
        
        return results
    
    def _greedy_winner_determination(self, bids: List[Dict], tasks: List) -> Dict:
        """Determine winners using a greedy algorithm."""
        # Sort bids by value-per-task (descending)
        sorted_bids = sorted(bids, key=lambda b: b['bid_value'] / len(b['task_ids']), reverse=True)
        
        # Initialize allocation
        allocation = {}  # task_id -> robot_id
        allocated_tasks = set()
        
        # Allocate bids in order
        for bid in sorted_bids:
            robot_id = bid['robot_id']
            task_ids = bid['task_ids']
            
            # Check if all tasks in this bid are still available
            if all(task_id not in allocated_tasks for task_id in task_ids):
                # Allocate all tasks in this bid to the robot
                for task_id in task_ids:
                    allocation[task_id] = robot_id
                    allocated_tasks.add(task_id)
        
        return allocation
    
    def _vcg_winner_determination(self, bids: List[Dict], tasks: List) -> Tuple[Dict, Dict]:
        """Determine winners and payments using the VCG mechanism."""
        # For simplicity, use a timeout to avoid excessive computation
        timeout = min(self.timeout, 30.0)  # Cap at 30 seconds max
        start_time = time.time()
        
        # If the problem is too large, fall back to greedy
        if len(tasks) > 15 or len(bids) > 50:
            print("Problem too large for optimal VCG, using greedy approximation")
            allocation = self._greedy_winner_determination(bids, tasks)
            payments = self._calculate_vcg_payments_approx(allocation, bids, tasks)
            return allocation, payments
        
        # Step 1: Find optimal allocation (maximizing social welfare)
        allocation, welfare = self._find_optimal_allocation(bids, tasks, timeout)
        
        # Step 2: Calculate VCG payments
        payments = {}
        task_ids = [task.id for task in tasks]
        
        for task_id in allocation:
            # Calculate welfare without this task
            tasks_without_i = [t for t in tasks if t.id != task_id]
            bids_without_i = [b for b in bids if task_id not in b['task_ids']]
            
            # Find optimal allocation without task i
            allocation_without_i, welfare_without_i = self._find_optimal_allocation(
                bids_without_i, tasks_without_i, max(0.1, timeout / len(allocation)))
            
            # Calculate payment: difference in others' welfare
            # First, sum of others' welfare in optimal allocation
            winner_robot_id = allocation[task_id]
            others_welfare = welfare
            for t_id, r_id in allocation.items():
                if r_id == winner_robot_id:
                    # Find original bid value for this task
                    for bid in bids:
                        if bid['robot_id'] == r_id and t_id in bid['task_ids']:
                            if len(bid['task_ids']) == 1:
                                others_welfare -= bid['bid_value']
                            else:
                                # For bundle bids, we need a more complex approach
                                # Here we approximate by proportional allocation
                                others_welfare -= bid['bid_value'] / len(bid['task_ids'])
            
            # Then, welfare of others in allocation without i
            payment = welfare_without_i - others_welfare
            payments[task_id] = max(0, payment)  # Ensure non-negative payment
        
        return allocation, payments
    
    def _find_optimal_allocation(self, bids: List[Dict], tasks: List, timeout: float) -> Tuple[Dict, float]:
        """Find the allocation that maximizes social welfare."""
        # Convert to task IDs for easier processing
        task_ids = [task.id for task in tasks]
        
        # Setup for branch and bound
        start_time = time.time()
        best_allocation = {}
        best_welfare = 0
        
        # Create a graph representation to speed up conflict checking
        task_to_bids = defaultdict(list)
        for i, bid in enumerate(bids):
            for task_id in bid['task_ids']:
                task_to_bids[task_id].append(i)
        
        # Helper function to check if two bids conflict
        def bids_conflict(bid1, bid2):
            return any(task_id in bid2['task_ids'] for task_id in bid1['task_ids'])
        
        # Helper function for recursive branch and bound
        def branch_and_bound(current_allocation, current_welfare, remaining_bids):
            nonlocal best_allocation, best_welfare
            
            # Check timeout
            if time.time() - start_time > timeout:
                return
            
            # Update best solution if current is better
            if current_welfare > best_welfare:
                best_allocation = current_allocation.copy()
                best_welfare = current_welfare
            
            # No more bids to consider
            if not remaining_bids:
                return
            
            # Get the next bid
            current_bid = remaining_bids[0]
            next_remaining = remaining_bids[1:]
            
            # Option 1: Skip this bid
            branch_and_bound(current_allocation, current_welfare, next_remaining)
            
            # Option 2: Include this bid if it doesn't conflict
            can_include = True
            for task_id in current_bid['task_ids']:
                if task_id in current_allocation:
                    can_include = False
                    break
            
            if can_include:
                # Add this bid to allocation
                new_allocation = current_allocation.copy()
                for task_id in current_bid['task_ids']:
                    new_allocation[task_id] = current_bid['robot_id']
                
                new_welfare = current_welfare + current_bid['bid_value']
                
                # Recursively explore
                branch_and_bound(new_allocation, new_welfare, next_remaining)
        
        # If problem is manageable, use branch and bound
        if len(bids) <= 20:
            branch_and_bound({}, 0, bids)
        else:
            # For larger problems, use greedy approach
            best_allocation = self._greedy_winner_determination(bids, tasks)
            
            # Calculate welfare
            allocated_tasks = set(best_allocation.keys())
            for bid in bids:
                if all(task_id in allocated_tasks and best_allocation[task_id] == bid['robot_id'] 
                       for task_id in bid['task_ids']):
                    best_welfare += bid['bid_value']
        
        return best_allocation, best_welfare
    
    def _calculate_vcg_payments_approx(self, allocation: Dict, bids: List[Dict], tasks: List) -> Dict:
        """Calculate approximate VCG payments when optimal solution is too expensive."""
        payments = {}
        
        # Create map of task to bid value
        task_bid_values = defaultdict(dict)
        for bid in bids:
            robot_id = bid['robot_id']
            
            # Handle single-task bids
            if len(bid['task_ids']) == 1:
                task_id = bid['task_ids'][0]
                task_bid_values[task_id][robot_id] = bid['bid_value']
            else:
                # Approximate value per task in bundle
                value_per_task = bid['bid_value'] / len(bid['task_ids'])
                for task_id in bid['task_ids']:
                    task_bid_values[task_id][robot_id] = value_per_task
        
        # Calculate payments task by task
        for task_id, winner_id in allocation.items():
            # Get all bids for this task
            bids_for_task = task_bid_values.get(task_id, {})
            
            # Find the highest bid from non-winners (second price)
            second_price = self.reserve_price
            for robot_id, bid_value in bids_for_task.items():
                if robot_id != winner_id and bid_value > second_price:
                    second_price = bid_value
            
            payments[task_id] = second_price
        
        return payments
    
    def _calculate_payments(self, allocation: Dict, bids: List[Dict]) -> Dict:
        """Calculate payments based on payment rule."""
        payments = {}
        
        if self.payment_rule == 'first_price':
            # First price: winners pay their bid
            # Create a map of robot's bid for each task
            task_to_bid = {}
            
            for bid in bids:
                robot_id = bid['robot_id']
                task_ids = bid['task_ids']
                bid_value = bid['bid_value']
                
                # Distribute bid value across tasks for bundle bids
                if len(task_ids) > 1:
                    value_per_task = bid_value / len(task_ids)
                    for task_id in task_ids:
                        if allocation.get(task_id) == robot_id:
                            task_to_bid[task_id] = value_per_task
                else:
                    # Single task bid
                    task_id = task_ids[0]
                    if allocation.get(task_id) == robot_id:
                        task_to_bid[task_id] = bid_value
            
            # Set payments
            for task_id, robot_id in allocation.items():
                payments[task_id] = task_to_bid.get(task_id, self.reserve_price)
                
        elif self.payment_rule == 'second_price':
            # Second price: task-by-task second highest bid
            # For bundles, this is an approximation
            
            # Create map of all bids per task
            task_bids = defaultdict(list)
            
            for bid in bids:
                robot_id = bid['robot_id']
                
                if len(bid['task_ids']) == 1:
                    # Single task bid - straightforward
                    task_id = bid['task_ids'][0]
                    task_bids[task_id].append((robot_id, bid['bid_value']))
                else:
                    # Bundle bid - distribute value proportionally
                    value_per_task = bid['bid_value'] / len(bid['task_ids'])
                    for task_id in bid['task_ids']:
                        task_bids[task_id].append((robot_id, value_per_task))
            
            # For each allocated task, find second highest bid
            for task_id, robot_id in allocation.items():
                bids_for_task = task_bids.get(task_id, [])
                
                # Sort bids (highest first)
                sorted_bids = sorted(bids_for_task, key=lambda x: x[1], reverse=True)
                
                # Find highest bid from a non-winner
                second_price = self.reserve_price
                for bid_robot_id, bid_value in sorted_bids:
                    if bid_robot_id != robot_id and bid_value > second_price:
                        second_price = bid_value
                        break
                
                payments[task_id] = second_price
                
        else:
            # Default to first price if rule not recognized
            for task_id in allocation:
                payments[task_id] = self.reserve_price
                
        return payments
    
    def allocate_tasks(self, winners: Dict, tasks: List, robots: List) -> Dict:
        """
        Finalize task allocation based on auction results.
        
        Args:
            winners: Winner determination results
            tasks: List of tasks being allocated
            robots: List of robots participating in the auction
            
        Returns:
            Dict: Final allocation results and statistics
        """
        # Map task and robot objects by ID for easy lookup
        task_map = {task.id: task for task in tasks}
        robot_map = {robot.id: robot for robot in robots}
        
        allocations = winners.get('allocations', {})
        payments = winners.get('payments', {})
        unallocated = winners.get('unallocated', [])
        
        # Results structure
        results = {
            'allocated_tasks': [],
            'unallocated_tasks': [],
            'robot_allocations': defaultdict(list),
            'payments': {},
            'metrics': {
                'allocation_rate': 0,
                'total_welfare': winners.get('welfare', 0),
                'total_revenue': sum(payments.values())
            }
        }
        
        # Process allocations
        for task_id, robot_id in allocations.items():
            task = task_map.get(task_id)
            robot = robot_map.get(robot_id)
            
            if task and robot:
                # Update task status
                task.assigned = True
                task.assigned_to = robot_id
                
                # Update robot task list
                if hasattr(robot, 'current_tasks'):
                    robot.current_tasks.add(task_id)
                
                # Record allocation
                results['allocated_tasks'].append(task)
                results['robot_allocations'][robot_id].append(task)
                results['payments'][task_id] = payments.get(task_id, 0)
                
                # Notify robots of the outcome
                if hasattr(robot, 'update_strategy'):
                    auction_results = {
                        task_id: {
                            'winner': robot_id,
                            'payment': payments.get(task_id, 0),
                            'type': task.type,
                            'true_utility': getattr(robot, 'evaluate_task', lambda x: 0)(task)
                        }
                    }
                    robot.update_strategy(auction_results)
        
        # Record unallocated tasks
        for task_id in unallocated:
            task = task_map.get(task_id)
            if task:
                results['unallocated_tasks'].append(task)
        
        # Calculate final metrics
        total_tasks = len(tasks)
        allocated_count = len(results['allocated_tasks'])
        results['metrics']['allocation_rate'] = allocated_count / total_tasks if total_tasks > 0 else 0
        
        # Message count for allocation
        self.metrics['message_count'][-1] += allocated_count  # One message per allocation
        
        return results
    
    def evaluate_allocation_efficiency(self, allocation: Dict, optimal_allocation: Dict = None) -> Dict:
        """
        Evaluate the efficiency of the allocation compared to an optimal allocation.
        
        Args:
            allocation: The actual allocation from the auction
            optimal_allocation: An optimal allocation (if available)
            
        Returns:
            Dict: Efficiency metrics
        """
        # Calculate actual welfare
        actual_welfare = allocation['metrics']['total_welfare']
        
        # If optimal allocation is not provided, use stats from actual allocation
        if optimal_allocation is None:
            return {
                'allocation_rate': allocation['metrics']['allocation_rate'],
                'actual_welfare': actual_welfare,
                'optimal_welfare': actual_welfare,  # Assume actual is optimal
                'welfare_ratio': 1.0,
                'revenue': allocation['metrics']['total_revenue']
            }
        
        # Calculate optimal welfare
        optimal_welfare = optimal_allocation['metrics']['total_welfare']
        
        # Calculate efficiency metrics
        welfare_ratio = actual_welfare / optimal_welfare if optimal_welfare > 0 else 0
        allocation_rate_optimal = optimal_allocation['metrics']['allocation_rate']
        
        return {
            'allocation_rate': allocation['metrics']['allocation_rate'],
            'allocation_rate_optimal': allocation_rate_optimal,
            'actual_welfare': actual_welfare,
            'optimal_welfare': optimal_welfare,
            'welfare_ratio': welfare_ratio,
            'revenue': allocation['metrics']['total_revenue'],
            'revenue_optimal': optimal_allocation['metrics'].get('total_revenue', 0)
        }
    
    def visualize_auction_process(self, bids: Dict, winners: Dict, payments: Dict = None) -> plt.Figure:
        """
        Create a visualization of the auction process, including bids and results.
        
        Args:
            bids: Collected bids
            winners: Winner determination results
            payments: Payment information
            
        Returns:
            matplotlib.figure.Figure: Figure with visualization
        """
        fig = plt.figure(figsize=(12, 8))
        
        # Different visualizations based on auction type
        if self.auction_type in ['sequential', 'parallel']:
            # For sequential/parallel auctions: bar chart of bids per task
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.set_title(f"{self.auction_type.capitalize()} Auction with {self.payment_rule} payment rule")
            
            # Organize data for plotting
            task_ids = sorted(bids.keys())
            x_positions = range(len(task_ids))
            winning_bids = []
            second_highest = []
            all_bids = []
            
            for task_id in task_ids:
                task_bids = bids.get(task_id, [])
                sorted_bids = sorted([b['bid_value'] for b in task_bids], reverse=True)
                
                winning_bids.append(sorted_bids[0] if sorted_bids else 0)
                second_highest.append(sorted_bids[1] if len(sorted_bids) > 1 else 0)
                all_bids.append(sorted_bids)
            
            # Bar chart of winning and second-highest bids
            ax1.bar(x_positions, winning_bids, width=0.4, label='Winning Bid')
            ax1.bar([x + 0.4 for x in x_positions], second_highest, width=0.4, label='Second Highest')
            
            # Add payment bars
            if payments:
                payment_values = [payments.get(task_id, 0) for task_id in task_ids]
                ax1.bar([x + 0.2 for x in x_positions], payment_values, width=0.2, color='red', label='Payment')
            
            # Add bid distributions as scatter points
            for i, bids_list in enumerate(all_bids):
                if len(bids_list) > 2:
                    ax1.scatter([i] * (len(bids_list) - 2), bids_list[2:], color='gray', alpha=0.5)
            
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels([f'Task {tid}' for tid in task_ids])
            ax1.set_ylabel('Bid Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot allocation metrics
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.set_title('Auction Performance Metrics')
            
            metrics = ['social_welfare', 'revenue', 'allocation_efficiency', 'execution_time']
            metric_values = [self.metrics[m][-1] if self.metrics[m] else 0 for m in metrics]
            
            ax2.bar(metrics, metric_values)
            ax2.set_ylabel('Value')
            ax2.grid(True, alpha=0.3)
            
        elif self.auction_type == 'combinatorial':
            # For combinatorial auctions: network diagram of bundle allocations
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.set_title(f"Combinatorial Auction with {self.payment_rule} payment rule")
            
            # Create a graph where nodes are tasks and robots, edges are allocations
            G = nx.Graph()
            
            # Get allocations
            allocations = winners.get('allocations', {})
            
            # Add task nodes
            for task_id in allocations:
                G.add_node(f"Task {task_id}", type='task')
            
            # Add robot nodes and edges
            for task_id, robot_id in allocations.items():
                robot_name = f"Robot {robot_id}"
                if robot_name not in G:
                    G.add_node(robot_name, type='robot')
                G.add_edge(robot_name, f"Task {task_id}")
            
            # Position nodes using spring layout
            pos = nx.spring_layout(G)
            
            # Draw the graph
            task_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'task']
            robot_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'robot']
            
            nx.draw_networkx_nodes(G, pos, nodelist=task_nodes, node_color='lightblue', 
                                 node_size=300, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, nodelist=robot_nodes, node_color='lightgreen', 
                                 node_size=500, alpha=0.8)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            nx.draw_networkx_labels(G, pos, font_size=10)
            
            # Remove axis
            ax1.axis('off')
            
            # Plot bundle distribution
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.set_title('Bundle Size Distribution')
            
            # Count allocated bundles by size
            robot_bundles = defaultdict(list)
            for task_id, robot_id in allocations.items():
                robot_bundles[robot_id].append(task_id)
            
            bundle_sizes = [len(tasks) for tasks in robot_bundles.values()]
            size_counts = defaultdict(int)
            for size in bundle_sizes:
                size_counts[size] += 1
            
            sizes = sorted(size_counts.keys())
            counts = [size_counts[s] for s in sizes]
            
            ax2.bar(sizes, counts)
            ax2.set_xlabel('Bundle Size')
            ax2.set_ylabel('Count')
            ax2.set_xticks(sizes)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_auction_metrics(self) -> Dict:
        """
        Get performance metrics for all auctions run by this mechanism.
        
        Returns:
            Dict: Summary statistics of auction performance
        """
        metrics = {}
        
        for key in self.metrics:
            values = self.metrics[key]
            if values:
                metrics[f'avg_{key}'] = sum(values) / len(values)
                metrics[f'min_{key}'] = min(values)
                metrics[f'max_{key}'] = max(values)
            else:
                metrics[f'avg_{key}'] = 0
                metrics[f'min_{key}'] = 0
                metrics[f'max_{key}'] = 0
        
        # Additional derived metrics
        if self.metrics['social_welfare'] and self.metrics['revenue']:
            # Ratio of revenue to welfare
            revenue_ratio = [r / w if w > 0 else 0 
                            for r, w in zip(self.metrics['revenue'], self.metrics['social_welfare'])]
            
            metrics['avg_revenue_ratio'] = sum(revenue_ratio) / len(revenue_ratio) if revenue_ratio else 0
            metrics['min_revenue_ratio'] = min(revenue_ratio) if revenue_ratio else 0
            metrics['max_revenue_ratio'] = max(revenue_ratio) if revenue_ratio else 0
        
        # Count of auctions
        metrics['auction_count'] = len(self.auction_history)
        
        return metrics


# Example usage
if __name__ == "__main__":
    # Import Task and BiddingRobot classes for testing
    from task_manager import Task, TaskManager
    from bidding_robot import BiddingRobot
    
    # Create a task manager
    task_manager = TaskManager(
        environment_size=(50, 50),
        task_complexity_range=(1.0, 5.0)
    )
    
    # Generate some tasks
    tasks = task_manager.generate_task_batch(10, include_complex=True)
    
    # Create some robots with different strategies
    robots = [
        BiddingRobot(0, {"speed": 1.5, "lift_capacity": 2.0}, (10, 10), "truthful"),
        BiddingRobot(1, {"speed": 1.0, "precision": 1.8}, (20, 20), "strategic"),
        BiddingRobot(2, {"lift_capacity": 1.2, "sensor_range": 2.0}, (30, 30), "cooperative"),
        BiddingRobot(3, {"speed": 1.3, "precision": 1.5, "sensor_range": 1.7}, (40, 40), "learning")
    ]
    
    # Create auction mechanisms
    sequential_auction = AuctionMechanism(auction_type="sequential", payment_rule="second_price")
    combinatorial_auction = AuctionMechanism(auction_type="combinatorial", payment_rule="first_price")
    
    # Run sequential auction
    print("Running sequential auction...")
    announcement = sequential_auction.announce_tasks(tasks)
    bids = sequential_auction.collect_bids(robots, tasks, announcement)
    winners = sequential_auction.determine_winners(bids, tasks)
    allocation = sequential_auction.allocate_tasks(winners, tasks, robots)
    
    print(f"Sequential auction results:")
    print(f"  Allocated tasks: {len(allocation['allocated_tasks'])}")
    print(f"  Unallocated tasks: {len(allocation['unallocated_tasks'])}")
    print(f"  Total welfare: {allocation['metrics']['total_welfare']:.2f}")
    print(f"  Total revenue: {allocation['metrics']['total_revenue']:.2f}")
    
    # Run combinatorial auction
    print("\nRunning combinatorial auction...")
    announcement = combinatorial_auction.announce_tasks(tasks)
    bids = combinatorial_auction.collect_bids(robots, tasks, announcement)
    winners = combinatorial_auction.determine_winners(bids, tasks)
    allocation = combinatorial_auction.allocate_tasks(winners, tasks, robots)
    
    print(f"Combinatorial auction results:")
    print(f"  Allocated tasks: {len(allocation['allocated_tasks'])}")
    print(f"  Unallocated tasks: {len(allocation['unallocated_tasks'])}")
    print(f"  Total welfare: {allocation['metrics']['total_welfare']:.2f}")
    print(f"  Total revenue: {allocation['metrics']['total_revenue']:.2f}")
    
    # Compare metrics
    sequential_metrics = sequential_auction.get_auction_metrics()
    combinatorial_metrics = combinatorial_auction.get_auction_metrics()
    
    print("\nPerformance comparison:")
    print(f"  Sequential allocation efficiency: {sequential_metrics['avg_allocation_efficiency']:.2f}")
    print(f"  Combinatorial allocation efficiency: {combinatorial_metrics['avg_allocation_efficiency']:.2f}")
    print(f"  Sequential social welfare: {sequential_metrics['avg_social_welfare']:.2f}")
    print(f"  Combinatorial social welfare: {combinatorial_metrics['avg_social_welfare']:.2f}")
    
    # Visualize results
    fig1 = sequential_auction.visualize_auction_process(bids, winners, winners.get('payments', {}))
    plt.figure(fig1.number)
    plt.savefig("sequential_auction_results.png")
    
    fig2 = combinatorial_auction.visualize_auction_process(bids, winners, winners.get('payments', {}))
    plt.figure(fig2.number)
    plt.savefig("combinatorial_auction_results.png")
    
    plt.show()