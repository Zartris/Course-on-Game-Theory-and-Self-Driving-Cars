"""
Fault-Tolerant Mechanism Design for Multi-Robot Systems

This module implements a fault-tolerant task allocation mechanism that maintains
robustness to robot failures, communication disruptions, and Byzantine behavior.
"""

import time
import random
import math
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """Status of a task in the allocation process."""
    UNALLOCATED = 0
    ALLOCATED = 1
    IN_PROGRESS = 2
    COMPLETED = 3
    FAILED = 4


@dataclass
class Task:
    """Representation of a task to be allocated."""
    id: str
    position: Tuple[float, float]  # (x, y) coordinates
    difficulty: float  # 0.0 to 1.0
    deadline: float  # seconds from now
    value: float
    required_capabilities: Set[str]
    status: TaskStatus = TaskStatus.UNALLOCATED
    assigned_robot: Optional[str] = None


@dataclass
class Robot:
    """Representation of a robot in the system."""
    id: str
    position: Tuple[float, float]  # (x, y) coordinates
    capabilities: Set[str]
    reliability: float  # 0.0 to 1.0
    battery_level: float  # 0.0 to 1.0
    is_byzantine: bool = False
    is_failed: bool = False
    last_heartbeat: float = 0.0


class FaultTolerantTaskAllocation:
    """
    Implements a fault-tolerant task allocation mechanism with Byzantine robustness,
    redundancy, and failure recovery.
    """
    
    def __init__(self, 
                 robots: List[Robot], 
                 tasks: List[Task], 
                 byzantine_threshold: int = 0,
                 heartbeat_timeout: float = 30.0,
                 redundancy_level: int = 1):
        """
        Initialize the fault-tolerant task allocation mechanism.
        
        Args:
            robots: List of robots in the system
            tasks: List of tasks to be allocated
            byzantine_threshold: Maximum number of Byzantine robots to tolerate
            heartbeat_timeout: Time in seconds after which a robot is considered failed
            redundancy_level: Number of backup robots to assign to each task
        """
        self.robots = {robot.id: robot for robot in robots}
        self.tasks = {task.id: task for task in tasks}
        self.byzantine_threshold = byzantine_threshold
        self.heartbeat_timeout = heartbeat_timeout
        self.redundancy_level = redundancy_level
        
        # Allocation data structures
        self.bids = {}  # task_id -> {robot_id -> bid_value}
        self.allocations = {}  # task_id -> [primary_robot_id, backup1_id, ...]
        self.payments = {}  # robot_id -> payment_amount
        
        # Fault tolerance data structures
        self.signatures = {}  # message_id -> {robot_id -> signature}
        self.failed_robots = set()  # Set of failed robot IDs
        self.message_history = {}  # robot_id -> last_message_time
        self.allocation_confirmations = {}  # task_id -> {robot_id -> confirmed}
        
        # Initialize heartbeat timestamps
        current_time = time.time()
        for robot_id in self.robots:
            self.robots[robot_id].last_heartbeat = current_time
            self.message_history[robot_id] = current_time
    
    def required_quorum(self) -> int:
        """
        Calculate required quorum size for Byzantine tolerance.
        
        Returns:
            int: Minimum number of robots required for a valid quorum
        """
        return 2 * self.byzantine_threshold + 1
    
    def sign_message(self, message: Dict[str, Any]) -> str:
        """
        Sign a message with the robot's cryptographic key.
        
        Args:
            message: Message to sign
            
        Returns:
            str: Signature for the message
        """
        # In a real system, this would use proper cryptographic signatures
        # For this example, we'll use a simple hash-based approach
        message_str = str(sorted(message.items()))
        return f"SIG_{hash(message_str) % 10000:04d}"
    
    def verify_signature(self, message: Dict[str, Any], robot_id: str, signature: str) -> bool:
        """
        Verify the signature on a message.
        
        Args:
            message: Message that was signed
            robot_id: ID of the robot that signed the message
            signature: Signature to verify
            
        Returns:
            bool: True if signature is valid, False otherwise
        """
        # In a real system, this would use proper cryptographic verification
        # For this example, we'll just check our simple hash-based signatures
        expected_sig = self.sign_message(message)
        
        # If the robot is Byzantine, it might produce invalid signatures
        if self.robots[robot_id].is_byzantine:
            # Byzantine robots have a chance of producing invalid signatures
            if random.random() < 0.3:
                return False
        
        return signature == expected_sig
    
    def generate_message_id(self, message: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a message.
        
        Args:
            message: Message to generate ID for
            
        Returns:
            str: Unique message ID
        """
        # In a real system, this would use a more sophisticated approach
        message_str = str(sorted(message.items()))
        return f"MSG_{hash(message_str) % 10000:04d}"
    
    def submit_bid(self, robot_id: str, task_id: str, bid_value: float) -> Tuple[Dict[str, Any], str, str]:
        """
        Submit a signed bid for a task.
        
        Args:
            robot_id: ID of the robot submitting the bid
            task_id: ID of the task being bid on
            bid_value: Value of the bid
            
        Returns:
            Tuple containing the message, message ID, and signature
        """
        # Create bid message
        message = {
            'type': 'bid',
            'task_id': task_id,
            'robot_id': robot_id,
            'bid_value': bid_value,
            'timestamp': time.time()
        }
        
        # Sign the message
        signature = self.sign_message(message)
        message_id = self.generate_message_id(message)
        
        return message, message_id, signature
    
    def process_bid(self, message: Dict[str, Any], signature: str) -> bool:
        """
        Process a bid message with signature verification.
        
        Args:
            message: Bid message
            signature: Signature on the message
            
        Returns:
            bool: True if bid was processed successfully, False otherwise
        """
        # Extract message fields
        robot_id = message.get('robot_id')
        task_id = message.get('task_id')
        bid_value = message.get('bid_value')
        
        # Verify signature
        if not self.verify_signature(message, robot_id, signature):
            print(f"Invalid signature from robot {robot_id}")
            return False
        
        # Check if robot is known to have failed
        if robot_id in self.failed_robots:
            print(f"Ignoring bid from failed robot {robot_id}")
            return False
        
        # Update heartbeat timestamp
        self.robots[robot_id].last_heartbeat = time.time()
        self.message_history[robot_id] = time.time()
        
        # Initialize bid structure for this task if needed
        if task_id not in self.bids:
            self.bids[task_id] = {}
        
        # Record the bid
        self.bids[task_id][robot_id] = bid_value
        
        print(f"Processed bid from robot {robot_id} for task {task_id}: {bid_value}")
        return True
    
    def determine_allocation(self, task_id: str) -> Optional[List[str]]:
        """
        Determine task allocation with redundancy.
        
        Args:
            task_id: ID of the task to allocate
            
        Returns:
            List of robot IDs (primary and backups) or None if allocation not possible
        """
        # Check if we have enough bids for a Byzantine-robust decision
        if task_id not in self.bids or len(self.bids[task_id]) < self.required_quorum():
            print(f"Not enough bids for task {task_id} to make a Byzantine-robust decision")
            return None
        
        # Sort robots by bid value (ascending for cost minimization)
        sorted_bids = sorted(self.bids[task_id].items(), key=lambda x: x[1])
        
        # Get required number of robots (primary + backups)
        num_robots_needed = 1 + self.redundancy_level
        
        # Check if we have enough robots
        if len(sorted_bids) < num_robots_needed:
            print(f"Not enough robots for task {task_id} with required redundancy")
            return None
        
        # Allocate to the best robots (primary and backups)
        allocation = [robot_id for robot_id, _ in sorted_bids[:num_robots_needed]]
        
        return allocation
    
    def broadcast_allocation(self, task_id: str, allocation: List[str]) -> Tuple[Dict[str, Any], str, str]:
        """
        Broadcast a task allocation with signature.
        
        Args:
            task_id: ID of the task being allocated
            allocation: List of robot IDs (primary and backups)
            
        Returns:
            Tuple containing the message, message ID, and signature
        """
        # Create allocation message
        message = {
            'type': 'allocation',
            'task_id': task_id,
            'allocation': allocation,
            'timestamp': time.time()
        }
        
        # Sign the message
        signature = self.sign_message(message)
        message_id = self.generate_message_id(message)
        
        return message, message_id, signature
    
    def process_allocation(self, message: Dict[str, Any], signature: str, sender_id: str) -> bool:
        """
        Process an allocation message with Byzantine consensus.
        
        Args:
            message: Allocation message
            signature: Signature on the message
            sender_id: ID of the robot that sent the message
            
        Returns:
            bool: True if allocation was processed successfully, False otherwise
        """
        # Extract message fields
        task_id = message.get('task_id')
        allocation = message.get('allocation')
        
        # Verify signature
        if not self.verify_signature(message, sender_id, signature):
            print(f"Invalid allocation signature from robot {sender_id}")
            return False
        
        # Update heartbeat timestamp
        self.robots[sender_id].last_heartbeat = time.time()
        self.message_history[sender_id] = time.time()
        
        # Generate message ID
        message_id = self.generate_message_id(message)
        
        # Initialize signature collection for this message if needed
        if message_id not in self.signatures:
            self.signatures[message_id] = {}
        
        # Record the signature
        self.signatures[message_id][sender_id] = signature
        
        # Initialize allocation confirmation for this task if needed
        if task_id not in self.allocation_confirmations:
            self.allocation_confirmations[task_id] = {}
        
        # Record confirmation
        self.allocation_confirmations[task_id][sender_id] = True
        
        # Check if we have enough signatures for Byzantine consensus
        if len(self.signatures[message_id]) >= self.required_quorum():
            # Accept the allocation
            self.allocations[task_id] = allocation
            
            # Update task status
            self.tasks[task_id].status = TaskStatus.ALLOCATED
            self.tasks[task_id].assigned_robot = allocation[0]  # Primary robot
            
            print(f"Task {task_id} allocated to {allocation} with Byzantine consensus")
            return True
        
        return False
    
    def detect_failures(self) -> List[str]:
        """
        Detect failed robots based on heartbeats.
        
        Returns:
            List of newly detected failed robot IDs
        """
        now = time.time()
        newly_failed = []
        
        # Check for missing heartbeats
        for robot_id, robot in self.robots.items():
            if robot_id not in self.failed_robots and now - robot.last_heartbeat > self.heartbeat_timeout:
                newly_failed.append(robot_id)
                self.failed_robots.add(robot_id)
                self.robots[robot_id].is_failed = True
                print(f"Robot {robot_id} detected as failed (no heartbeat for {self.heartbeat_timeout}s)")
        
        return newly_failed
    
    def adjust_allocation_for_failures(self, failed_robot_ids: List[str]) -> Dict[str, List[str]]:
        """
        Adjust allocations to account for failed robots.
        
        Args:
            failed_robot_ids: List of newly failed robot IDs
            
        Returns:
            Dict mapping task IDs to updated allocations
        """
        updated_allocations = {}
        
        for task_id, allocation in self.allocations.items():
            # Check if any of the allocated robots have failed
            if any(r_id in failed_robot_ids for r_id in allocation):
                # Filter out failed robots
                updated_allocation = [r_id for r_id in allocation if r_id not in self.failed_robots]
                
                # If primary has failed, promote backup
                if allocation[0] in failed_robot_ids and len(updated_allocation) > 0:
                    # Update task assignment
                    self.tasks[task_id].assigned_robot = updated_allocation[0]
                    
                    # Update allocation
                    self.allocations[task_id] = updated_allocation
                    updated_allocations[task_id] = updated_allocation
                    
                    print(f"Task {task_id}: Primary robot {allocation[0]} failed, promoted {updated_allocation[0]}")
                
                # If all assigned robots have failed, mark task as unallocated
                elif len(updated_allocation) == 0:
                    # Clear allocation to trigger reallocation
                    del self.allocations[task_id]
                    
                    # Reset task status
                    self.tasks[task_id].status = TaskStatus.UNALLOCATED
                    self.tasks[task_id].assigned_robot = None
                    
                    print(f"Task {task_id}: All assigned robots failed, marked for reallocation")
        
        return updated_allocations
    
    def calculate_vcg_payment(self, robot_id: str, task_id: str) -> float:
        """
        Calculate VCG payment for a robot.
        
        Args:
            robot_id: ID of the robot to calculate payment for
            task_id: ID of the task the robot is assigned to
            
        Returns:
            float: VCG payment amount
        """
        # If we don't have enough bids for this task, use a default payment
        if task_id not in self.bids or len(self.bids[task_id]) < 2:
            # Default to the robot's bid
            return self.bids.get(task_id, {}).get(robot_id, 0.0)
        
        # Get the robot's bid
        robot_bid = self.bids[task_id].get(robot_id, float('inf'))
        
        # Get the second-best bid (excluding the robot's bid)
        other_bids = [bid for r_id, bid in self.bids[task_id].items() if r_id != robot_id]
        second_best_bid = min(other_bids) if other_bids else robot_bid
        
        # VCG payment is the second-best bid
        return second_best_bid
    
    def run_allocation_round(self) -> Dict[str, Any]:
        """
        Run one round of the fault-tolerant allocation mechanism.
        
        Returns:
            Dict containing allocation results
        """
        # Detect and handle robot failures
        failed_robots = self.detect_failures()
        updated_allocations = {}
        
        if failed_robots:
            updated_allocations = self.adjust_allocation_for_failures(failed_robots)
        
        # Collect bids for unallocated tasks
        unallocated_tasks = [task_id for task_id, task in self.tasks.items() 
                            if task.status == TaskStatus.UNALLOCATED]
        
        new_allocations = {}
        
        for task_id in unallocated_tasks:
            # Determine allocation with redundancy
            allocation = self.determine_allocation(task_id)
            
            if allocation:
                # Broadcast allocation for consensus
                message, message_id, signature = self.broadcast_allocation(task_id, allocation)
                
                # Process our own allocation message
                if self.process_allocation(message, signature, list(self.robots.keys())[0]):
                    new_allocations[task_id] = allocation
                    
                    # Calculate payments for primary robot
                    primary_robot_id = allocation[0]
                    self.payments[primary_robot_id] = self.calculate_vcg_payment(primary_robot_id, task_id)
        
        # Return results
        return {
            'failed_robots': failed_robots,
            'updated_allocations': updated_allocations,
            'new_allocations': new_allocations,
            'payments': self.payments
        }


def simulate_fault_tolerant_allocation():
    """Run a simulation of the fault-tolerant task allocation mechanism."""
    # Create robots
    robots = [
        Robot(id="R1", position=(0, 0), capabilities={"sensing", "manipulation"}, 
              reliability=0.95, battery_level=0.8),
        Robot(id="R2", position=(10, 10), capabilities={"sensing", "locomotion"}, 
              reliability=0.9, battery_level=0.7),
        Robot(id="R3", position=(20, 20), capabilities={"manipulation", "locomotion"}, 
              reliability=0.85, battery_level=0.9),
        Robot(id="R4", position=(30, 30), capabilities={"sensing", "manipulation", "locomotion"}, 
              reliability=0.99, battery_level=0.6),
        Robot(id="R5", position=(40, 40), capabilities={"sensing"}, 
              reliability=0.8, battery_level=0.5, is_byzantine=True)
    ]
    
    # Create tasks
    tasks = [
        Task(id="T1", position=(5, 5), difficulty=0.3, deadline=300, value=100, 
             required_capabilities={"sensing"}),
        Task(id="T2", position=(15, 15), difficulty=0.5, deadline=200, value=150, 
             required_capabilities={"manipulation"}),
        Task(id="T3", position=(25, 25), difficulty=0.7, deadline=100, value=200, 
             required_capabilities={"sensing", "manipulation"}),
        Task(id="T4", position=(35, 35), difficulty=0.9, deadline=50, value=250, 
             required_capabilities={"locomotion", "manipulation"})
    ]
    
    # Initialize mechanism
    mechanism = FaultTolerantTaskAllocation(
        robots=robots,
        tasks=tasks,
        byzantine_threshold=1,
        redundancy_level=1
    )
    
    # Simulate bidding
    for robot in robots:
        for task in tasks:
            # Skip if robot doesn't have required capabilities
            if not task.required_capabilities.issubset(robot.capabilities):
                continue
            
            # Calculate distance to task
            distance = math.sqrt((robot.position[0] - task.position[0])**2 + 
                                (robot.position[1] - task.position[1])**2)
            
            # Calculate bid (cost) based on distance, task difficulty, and robot reliability
            base_cost = distance * task.difficulty
            reliability_factor = 1.0 / robot.reliability
            battery_factor = 1.0 / robot.battery_level
            
            bid_value = base_cost * reliability_factor * battery_factor
            
            # Byzantine robots may submit manipulated bids
            if robot.is_byzantine:
                if random.random() < 0.5:
                    # Underbid to win the task
                    bid_value *= 0.5
                else:
                    # Overbid to manipulate payments
                    bid_value *= 2.0
            
            # Submit bid
            message, message_id, signature = mechanism.submit_bid(robot.id, task.id, bid_value)
            mechanism.process_bid(message, signature)
    
    # Run allocation round
    results = mechanism.run_allocation_round()
    
    # Print results
    print("\nAllocation Results:")
    print(f"New Allocations: {results['new_allocations']}")
    print(f"Payments: {results['payments']}")
    
    # Simulate a robot failure
    print("\nSimulating failure of robot R2...")
    mechanism.robots["R2"].is_failed = True
    mechanism.failed_robots.add("R2")
    
    # Run another allocation round
    results = mechanism.run_allocation_round()
    
    # Print updated results
    print("\nUpdated Allocation Results:")
    print(f"Failed Robots: {results['failed_robots']}")
    print(f"Updated Allocations: {results['updated_allocations']}")
    print(f"New Allocations: {results['new_allocations']}")
    print(f"Payments: {results['payments']}")


if __name__ == "__main__":
    simulate_fault_tolerant_allocation()
