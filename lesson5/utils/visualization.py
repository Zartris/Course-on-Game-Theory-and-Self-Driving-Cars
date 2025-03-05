#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for the Task Allocation and Auction Mechanisms lesson.

This module provides functions for visualizing robots, tasks, auctions, and
allocation results in the multi-robot task allocation simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pygame
from typing import Dict, List, Tuple, Any, Optional
import networkx as nx
from matplotlib.patches import Rectangle, Circle, Arrow
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from collections import defaultdict

# Define colors for different elements
COLORS = {
    'robot': {
        'truthful': (0, 100, 255),      # Blue
        'strategic': (255, 100, 0),     # Orange
        'learning': (100, 180, 0),      # Green
        'cooperative': (180, 0, 180)    # Purple
    },
    'task': {
        'available': (0, 255, 0),       # Green
        'assigned': (0, 0, 255),        # Blue
        'completed': (100, 100, 100),   # Gray
        'failed': (255, 0, 0),          # Red
        'search': (0, 200, 100),        # Teal
        'transport': (200, 100, 0),     # Brown
        'monitor': (100, 100, 200),     # Light blue
        'rescue': (200, 0, 0),          # Red
        'build': (150, 150, 0)          # Olive
    },
    'coalition': (255, 0, 255),         # Magenta
    'bundle': (0, 200, 200),            # Cyan
    'communication': (255, 255, 0),     # Yellow
    'background': (240, 240, 240),      # Light gray
    'grid': (200, 200, 200),            # Medium gray
    'text': (0, 0, 0)                   # Black
}

def setup_pygame(width: int, height: int, title: str = "Task Allocation Simulation") -> Tuple[pygame.Surface, pygame.time.Clock]:
    """
    Initialize pygame and create a window.
    
    Args:
        width: Window width in pixels
        height: Window height in pixels
        title: Window title
        
    Returns:
        Tuple[pygame.Surface, pygame.time.Clock]: Pygame screen and clock objects
    """
    pygame.init()
    pygame.display.init()
    
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(title)
    clock = pygame.time.Clock()
    
    return screen, clock

def draw_grid(screen: pygame.Surface, grid_size: int, cell_size: int) -> None:
    """
    Draw a grid on the screen.
    
    Args:
        screen: Pygame screen
        grid_size: Number of cells in each dimension
        cell_size: Size of each cell in pixels
    """
    # Get screen dimensions
    width, height = screen.get_size()
    
    # Draw grid lines
    for x in range(0, width, cell_size):
        pygame.draw.line(screen, COLORS['grid'], (x, 0), (x, height), 1)
    for y in range(0, height, cell_size):
        pygame.draw.line(screen, COLORS['grid'], (0, y), (width, y), 1)
    
    # Draw grid coordinates (every 5 cells)
    font = pygame.font.Font(None, 18)
    for x in range(0, grid_size, 5):
        for y in range(0, grid_size, 5):
            text = font.render(f"{x},{y}", True, COLORS['text'])
            screen.blit(text, (x * cell_size + 2, y * cell_size + 2))

def draw_robots(screen: pygame.Surface, robots: List, cell_size: int, 
               selected_robot_id: Optional[int] = None) -> None:
    """
    Draw robots on the screen.
    
    Args:
        screen: Pygame screen
        robots: List of robot objects
        cell_size: Size of each cell in pixels
        selected_robot_id: ID of currently selected robot (highlighted)
    """
    for robot in robots:
        # Scale grid coordinates to pixel coordinates
        x, y = robot.position
        # Round for display purposes
        px = int(round(x) * cell_size + cell_size / 2)
        py = int(round(y) * cell_size + cell_size / 2)
        
        # Determine robot color based on strategy type
        color = COLORS['robot'].get(robot.strategy_type, (150, 150, 150))
        
        # Draw robot body
        radius = int(cell_size * 0.4)
        pygame.draw.circle(screen, color, (px, py), radius)
        
        # Draw robot ID
        font = pygame.font.Font(None, 22)
        id_text = font.render(str(robot.id), True, (255, 255, 255))
        id_rect = id_text.get_rect(center=(px, py))
        screen.blit(id_text, id_rect)
        
        # Draw highlight for selected robot
        if robot.id == selected_robot_id:
            pygame.draw.circle(screen, (255, 255, 0), (px, py), radius + 3, 2)
        
        # Draw energy level indicator
        if hasattr(robot, 'energy'):
            energy_ratio = robot.energy / 100.0  # Assuming max energy is 100
            energy_width = int(cell_size * 0.8)
            energy_height = int(cell_size * 0.1)
            energy_rect = pygame.Rect(
                px - energy_width // 2,
                py + radius + 2,
                int(energy_width * energy_ratio),
                energy_height
            )
            energy_outline = pygame.Rect(
                px - energy_width // 2,
                py + radius + 2,
                energy_width,
                energy_height
            )
            
            # Energy level color (green to red)
            energy_color = (
                int(255 * (1 - energy_ratio)),
                int(255 * energy_ratio),
                0
            )
            
            pygame.draw.rect(screen, (50, 50, 50), energy_outline, 1)
            pygame.draw.rect(screen, energy_color, energy_rect)
        
        # Draw assignment indicators (small dots for each assigned task)
        if hasattr(robot, 'current_tasks') and robot.current_tasks:
            n_tasks = len(robot.current_tasks)
            angle_step = 2 * np.pi / max(n_tasks, 1)
            indicator_radius = radius + 5
            
            for i, task_id in enumerate(robot.current_tasks):
                angle = i * angle_step
                ix = px + int(indicator_radius * np.cos(angle))
                iy = py + int(indicator_radius * np.sin(angle))
                pygame.draw.circle(screen, (255, 255, 255), (ix, iy), 3)

def draw_tasks(screen: pygame.Surface, tasks: List, cell_size: int, 
              selected_task_id: Optional[int] = None) -> None:
    """
    Draw tasks on the screen.
    
    Args:
        screen: Pygame screen
        tasks: List of task objects
        cell_size: Size of each cell in pixels
        selected_task_id: ID of currently selected task (highlighted)
    """
    # Group tasks by bundle
    bundle_tasks = defaultdict(list)
    for task in tasks:
        if hasattr(task, 'bundle_id') and task.bundle_id:
            bundle_tasks[task.bundle_id].append(task)
    
    # Draw bundle connections first (under tasks)
    for bundle_id, bundle_tasks_list in bundle_tasks.items():
        if len(bundle_tasks_list) > 1:
            # Connect bundle tasks with lines
            points = []
            for task in bundle_tasks_list:
                x, y = task.position
                px = int(x * cell_size + cell_size / 2)
                py = int(y * cell_size + cell_size / 2)
                points.append((px, py))
            
            # Draw lines connecting bundled tasks
            if len(points) >= 2:
                pygame.draw.lines(screen, COLORS['bundle'], False, points, 2)
    
    # Draw each task
    for task in tasks:
        # Scale grid coordinates to pixel coordinates
        x, y = task.position
        px = int(x * cell_size + cell_size / 2)
        py = int(y * cell_size + cell_size / 2)
        
        # Determine task color based on state and type
        if task.completed:
            color = COLORS['task']['completed']
        elif task.failed:
            color = COLORS['task']['failed']
        elif task.assigned:
            color = COLORS['task']['assigned']
        else:
            # Use task type color if available, otherwise default to available color
            color = COLORS['task'].get(task.type, COLORS['task']['available'])
        
        # Draw task shape (square for standard, diamond for complex, star for bundles)
        size = int(cell_size * 0.35)
        if task.requires_coalition:
            # Draw diamond for coalition tasks
            points = [
                (px, py - size),
                (px + size, py),
                (px, py + size),
                (px - size, py)
            ]
            pygame.draw.polygon(screen, color, points)
            # Add an outline
            pygame.draw.polygon(screen, COLORS['coalition'], points, 2)
        elif hasattr(task, 'bundle_id') and task.bundle_id:
            # Draw hexagon for bundled tasks
            points = []
            for i in range(6):
                angle = i * np.pi / 3
                tx = px + int(size * np.cos(angle))
                ty = py + int(size * np.sin(angle))
                points.append((tx, ty))
            pygame.draw.polygon(screen, color, points)
            # Add bundle outline
            pygame.draw.polygon(screen, COLORS['bundle'], points, 2)
        else:
            # Draw square for regular tasks
            rect = pygame.Rect(px - size, py - size, size * 2, size * 2)
            pygame.draw.rect(screen, color, rect)
        
        # Draw task ID
        font = pygame.font.Font(None, 20)
        id_text = font.render(str(task.id), True, (255, 255, 255))
        id_rect = id_text.get_rect(center=(px, py))
        screen.blit(id_text, id_rect)
        
        # Draw highlight for selected task
        if task.id == selected_task_id:
            highlight_size = size + 3
            if task.requires_coalition:
                highlight_points = [
                    (px, py - highlight_size),
                    (px + highlight_size, py),
                    (px, py + highlight_size),
                    (px - highlight_size, py)
                ]
                pygame.draw.polygon(screen, (255, 255, 0), highlight_points, 2)
            elif hasattr(task, 'bundle_id') and task.bundle_id:
                highlight_points = []
                for i in range(6):
                    angle = i * np.pi / 3
                    tx = px + int(highlight_size * np.cos(angle))
                    ty = py + int(highlight_size * np.sin(angle))
                    highlight_points.append((tx, ty))
                pygame.draw.polygon(screen, (255, 255, 0), highlight_points, 2)
            else:
                highlight_rect = pygame.Rect(
                    px - highlight_size, 
                    py - highlight_size, 
                    highlight_size * 2, 
                    highlight_size * 2
                )
                pygame.draw.rect(screen, (255, 255, 0), highlight_rect, 2)
        
        # Draw priority indicator (small triangle)
        if hasattr(task, 'priority'):
            priority_height = int(task.priority * 5)
            priority_points = [
                (px, py - size - priority_height),
                (px - 5, py - size - 2),
                (px + 5, py - size - 2)
            ]
            pygame.draw.polygon(screen, (255, 200, 0), priority_points)

def draw_allocations(screen: pygame.Surface, tasks: List, robots: List, 
                    allocations: Dict[int, int], cell_size: int) -> None:
    """
    Draw allocation links between robots and their assigned tasks.
    
    Args:
        screen: Pygame screen
        tasks: List of task objects
        robots: List of robot objects
        allocations: Dictionary mapping task IDs to robot IDs
        cell_size: Size of each cell in pixels
    """
    # Create maps for faster lookup
    task_map = {task.id: task for task in tasks}
    robot_map = {robot.id: robot for robot in robots}
    
    # Group allocations by robot
    robot_tasks = defaultdict(list)
    for task_id, robot_id in allocations.items():
        if task_id in task_map and robot_id in robot_map:
            robot_tasks[robot_id].append(task_id)
    
    # Draw allocation lines
    for robot_id, task_ids in robot_tasks.items():
        robot = robot_map[robot_id]
        rx, ry = robot.position
        # Round for display purposes
        display_rx, display_ry = round(rx), round(ry)
        rpx = int(display_rx * cell_size + cell_size / 2)
        rpy = int(display_ry * cell_size + cell_size / 2)
        
        for task_id in task_ids:
            task = task_map[task_id]
            tx, ty = task.position
            tpx = int(tx * cell_size + cell_size / 2)
            tpy = int(ty * cell_size + cell_size / 2)
            
            # Draw a line from robot to task
            color = COLORS['robot'].get(robot.strategy_type, (150, 150, 150))
            pygame.draw.line(screen, color, (rpx, rpy), (tpx, tpy), 1)
            
            # Draw an arrow near the task to indicate direction
            angle = np.arctan2(tpy - rpy, tpx - rpx)
            arrow_length = 10
            arrow_x = tpx - arrow_length * np.cos(angle)
            arrow_y = tpy - arrow_length * np.sin(angle)
            
            # Draw arrowhead
            size = 7
            arrow_points = [
                (tpx, tpy),
                (arrow_x + size * np.cos(angle + np.pi/2), arrow_y + size * np.sin(angle + np.pi/2)),
                (arrow_x + size * np.cos(angle - np.pi/2), arrow_y + size * np.sin(angle - np.pi/2))
            ]
            pygame.draw.polygon(screen, color, arrow_points)

def draw_auction_process(screen: pygame.Surface, auction_data: Dict, font: pygame.font.Font) -> None:
    """
    Draw auction process information on the screen.
    
    Args:
        screen: Pygame screen
        auction_data: Dictionary containing auction information
        font: Pygame font object for text rendering
    """
    # Get screen dimensions
    width, height = screen.get_size()
    
    # Create auction info panel in bottom right
    panel_width = 300
    panel_height = 200
    panel_x = width - panel_width - 10
    panel_y = height - panel_height - 10
    
    # Draw panel background
    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    pygame.draw.rect(screen, (220, 220, 220), panel_rect)
    pygame.draw.rect(screen, (100, 100, 100), panel_rect, 2)
    
    # Draw auction header
    auction_type = auction_data.get('auction_type', 'Unknown')
    payment_rule = auction_data.get('payment_rule', 'Unknown')
    header = f"Auction: {auction_type.capitalize()} ({payment_rule})"
    header_text = font.render(header, True, COLORS['text'])
    screen.blit(header_text, (panel_x + 10, panel_y + 10))
    
    # Draw auction metrics
    metrics = auction_data.get('metrics', {})
    y_offset = 40
    for i, (key, value) in enumerate(metrics.items()):
        if isinstance(value, (int, float)):
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            text_surface = font.render(text, True, COLORS['text'])
            screen.blit(text_surface, (panel_x + 10, panel_y + y_offset + i * 20))
    
    # Draw current phase
    phase = auction_data.get('current_phase', 'Idle')
    phase_text = font.render(f"Phase: {phase}", True, COLORS['text'])
    screen.blit(phase_text, (panel_x + 10, panel_y + panel_height - 30))

def draw_info_panel(screen: pygame.Surface, simulation_info: Dict, font: pygame.font.Font) -> None:
    """
    Draw simulation information panel on the screen.
    
    Args:
        screen: Pygame screen
        simulation_info: Dictionary containing simulation information
        font: Pygame font object for text rendering
    """
    # Get screen dimensions
    width, height = screen.get_size()
    
    # Create info panel in top right
    panel_width = 250
    panel_height = 180
    panel_x = width - panel_width - 10
    panel_y = 10
    
    # Draw panel background
    panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
    pygame.draw.rect(screen, (220, 220, 220), panel_rect)
    pygame.draw.rect(screen, (100, 100, 100), panel_rect, 2)
    
    # Draw simulation header
    header = "Simulation Info"
    header_text = font.render(header, True, COLORS['text'])
    screen.blit(header_text, (panel_x + 10, panel_y + 10))
    
    # Draw simulation metrics
    y_offset = 40
    for i, (key, value) in enumerate(simulation_info.items()):
        # Format based on value type
        if isinstance(value, (int, float)):
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
        else:
            text = f"{key}: {value}"
            
        text_surface = font.render(text, True, COLORS['text'])
        screen.blit(text_surface, (panel_x + 10, panel_y + y_offset + i * 20))

def create_animation(robots: List, tasks: List, auction_history: List, 
                    allocations: Dict, grid_size: int = 50,
                    duration: int = 10) -> animation.FuncAnimation:
    """
    Create an animation of the auction process and task allocation.
    
    Args:
        robots: List of robot objects
        tasks: List of task objects
        auction_history: List of auction events
        allocations: Final allocation mapping
        grid_size: Size of the grid
        duration: Animation duration in seconds
        
    Returns:
        matplotlib.animation.FuncAnimation: Animation object
    """
    # Setup figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_title('Task Allocation Animation')
    
    # Create task and robot patches
    task_patches = []
    for task in tasks:
        x, y = task.position
        
        if task.requires_coalition:
            # Diamond for coalition tasks
            size = 1.0
            patch = plt.Polygon([
                (x, y - size),
                (x + size, y),
                (x, y + size),
                (x - size, y)
            ], color='green', alpha=0.7)
        elif hasattr(task, 'bundle_id') and task.bundle_id:
            # Hexagon for bundled tasks
            size = 1.0
            points = []
            for i in range(6):
                angle = i * np.pi / 3
                tx = x + size * np.cos(angle)
                ty = y + size * np.sin(angle)
                points.append((tx, ty))
            patch = plt.Polygon(points, color='cyan', alpha=0.7)
        else:
            # Square for regular tasks
            patch = plt.Rectangle((x - 0.8, y - 0.8), 1.6, 1.6, color='green', alpha=0.7)
            
        task_patches.append(patch)
        ax.add_patch(patch)
        ax.text(x, y, str(task.id), ha='center', va='center', color='white', fontweight='bold')
    
    robot_patches = []
    for robot in robots:
        x, y = robot.position
        color = {
            'truthful': 'blue',
            'strategic': 'orange',
            'learning': 'green',
            'cooperative': 'purple'
        }.get(robot.strategy_type, 'gray')
        
        patch = plt.Circle((x, y), 1.0, color=color, alpha=0.7)
        robot_patches.append(patch)
        ax.add_patch(patch)
        ax.text(x, y, str(robot.id), ha='center', va='center', color='white', fontweight='bold')
    
    # Add grid lines
    for i in range(grid_size + 1):
        ax.axhline(i, color='gray', alpha=0.3)
        ax.axvline(i, color='gray', alpha=0.3)
    
    # Prepare allocation lines (initially invisible)
    allocation_lines = []
    for task_id, robot_id in allocations.items():
        task = next((t for t in tasks if t.id == task_id), None)
        robot = next((r for r in robots if r.id == robot_id), None)
        
        if task and robot:
            tx, ty = task.position
            rx, ry = robot.position
            line, = ax.plot([rx, tx], [ry, ty], 'r-', alpha=0, lw=1.5)
            allocation_lines.append((line, robot_id))
    
    # Animation function
    def update(frame):
        # Normalize frame to [0, 1] range
        t = frame / duration
        
        # Update task states based on auction history
        for i, task in enumerate(tasks):
            patch = task_patches[i]
            
            # Change color based on allocation state
            if task.id in allocations:
                if t > 0.5:  # Show allocations in second half of animation
                    patch.set_color('blue')
            elif task.failed:
                patch.set_color('red')
            
        # Reveal allocation lines progressively
        if t > 0.5:
            progress = (t - 0.5) * 2  # Scale to [0, 1] in second half
            for line, robot_id in allocation_lines:
                line.set_alpha(progress)
                # Get color based on robot type
                robot = next((r for r in robots if r.id == robot_id), None)
                if robot:
                    color = {
                        'truthful': 'blue',
                        'strategic': 'orange',
                        'learning': 'green',
                        'cooperative': 'purple'
                    }.get(robot.strategy_type, 'gray')
                    line.set_color(color)
        
        return task_patches + robot_patches + [line for line, _ in allocation_lines]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update, frames=np.linspace(0, duration, 60),
        interval=50, blit=True
    )
    
    return anim

def visualize_allocation_results(tasks: List, robots: List, allocations: Dict) -> plt.Figure:
    """
    Create a static visualization of allocation results.
    
    Args:
        tasks: List of task objects
        robots: List of robot objects
        allocations: Dictionary mapping task IDs to robot IDs
        
    Returns:
        matplotlib.figure.Figure: Figure with visualization
    """
    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    
    # Robot assignments subplot
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Task Assignments by Robot Type')
    
    # Count assignments by robot type
    robot_types = {}
    for robot in robots:
        if robot.strategy_type not in robot_types:
            robot_types[robot.strategy_type] = {'count': 0, 'tasks': 0}
        robot_types[robot.strategy_type]['count'] += 1
    
    for task_id, robot_id in allocations.items():
        robot = next((r for r in robots if r.id == robot_id), None)
        if robot:
            robot_types[robot.strategy_type]['tasks'] += 1
    
    # Create bar chart
    types = list(robot_types.keys())
    counts = [data['count'] for data in robot_types.values()]
    tasks = [data['tasks'] for data in robot_types.values()]
    
    x = np.arange(len(types))
    width = 0.35
    
    ax1.bar(x - width/2, counts, width, label='Robots')
    ax1.bar(x + width/2, tasks, width, label='Assigned Tasks')
    
    ax1.set_xlabel('Robot Type')
    ax1.set_ylabel('Count')
    ax1.set_xticks(x)
    ax1.set_xticklabels(types)
    ax1.legend()
    
    # Task allocation status subplot
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Task Allocation Status')
    
    # Count tasks by status
    task_status = {
        'Allocated': len([t for t in tasks if t.id in allocations]),
        'Unallocated': len([t for t in tasks if not t.assigned and not t.completed and not t.failed]),
        'Completed': len([t for t in tasks if t.completed]),
        'Failed': len([t for t in tasks if t.failed])
    }
    
    # Create pie chart
    status_labels = list(task_status.keys())
    status_counts = list(task_status.values())
    
    ax2.pie(status_counts, labels=status_labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax2.axis('equal')
    
    # Spatial distribution subplot
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('Spatial Distribution')
    
    # Plot tasks
    for task in tasks:
        x, y = task.position
        if task.id in allocations:
            color = 'blue'
        elif task.completed:
            color = 'gray'
        elif task.failed:
            color = 'red'
        else:
            color = 'green'
        
        marker = 's'  # Square for regular tasks
        if task.requires_coalition:
            marker = 'd'  # Diamond for coalition tasks
        elif hasattr(task, 'bundle_id') and task.bundle_id:
            marker = 'h'  # Hexagon for bundled tasks
            
        ax3.scatter(x, y, color=color, marker=marker, s=100, alpha=0.7)
        ax3.text(x, y, str(task.id), ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    
    # Plot robots
    for robot in robots:
        x, y = robot.position
        color = {
            'truthful': 'blue',
            'strategic': 'orange',
            'learning': 'green',
            'cooperative': 'purple'
        }.get(robot.strategy_type, 'gray')
        
        ax3.scatter(x, y, color=color, marker='o', s=150, alpha=0.7)
        ax3.text(x, y, str(robot.id), ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    
    # Draw allocation lines
    for task_id, robot_id in allocations.items():
        task = next((t for t in tasks if t.id == task_id), None)
        robot = next((r for r in robots if r.id == robot_id), None)
        
        if task and robot:
            tx, ty = task.position
            rx, ry = robot.position
            ax3.plot([rx, tx], [ry, ty], 'k-', alpha=0.5, lw=1)
    
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.grid(True, alpha=0.3)
    
    # Task type distribution subplot
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Task Type Distribution')
    
    # Count tasks by type
    task_types = {}
    for task in tasks:
        if task.type not in task_types:
            task_types[task.type] = {'total': 0, 'allocated': 0}
        task_types[task.type]['total'] += 1
        if task.id in allocations:
            task_types[task.type]['allocated'] += 1
    
    # Create stacked bar chart
    types = list(task_types.keys())
    allocated = [data['allocated'] for data in task_types.values()]
    unallocated = [data['total'] - data['allocated'] for data in task_types.values()]
    
    x = np.arange(len(types))
    
    ax4.bar(x, allocated, label='Allocated')
    ax4.bar(x, unallocated, bottom=allocated, label='Unallocated')
    
    ax4.set_xlabel('Task Type')
    ax4.set_ylabel('Count')
    ax4.set_xticks(x)
    ax4.set_xticklabels(types)
    ax4.legend()
    
    plt.tight_layout()
    return fig

def visualize_auction_comparison(auction_results: Dict) -> plt.Figure:
    """
    Create a comparison visualization of different auction types.
    
    Args:
        auction_results: Dictionary mapping auction types to result metrics
        
    Returns:
        matplotlib.figure.Figure: Figure with comparison
    """
    # Setup figure
    fig = plt.figure(figsize=(12, 8))
    
    # Social welfare comparison
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('Social Welfare Comparison')
    
    # Extract data
    auction_types = list(auction_results.keys())
    welfare = [results['welfare'] for results in auction_results.values()]
    
    ax1.bar(auction_types, welfare)
    ax1.set_xlabel('Auction Type')
    ax1.set_ylabel('Social Welfare')
    ax1.grid(True, alpha=0.3)
    
    # Revenue comparison
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title('Revenue Comparison')
    
    revenue = [results['revenue'] for results in auction_results.values()]
    
    ax2.bar(auction_types, revenue)
    ax2.set_xlabel('Auction Type')
    ax2.set_ylabel('Revenue')
    ax2.grid(True, alpha=0.3)
    
    # Efficiency comparison
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('Allocation Efficiency Comparison')
    
    # Calculate efficiency (allocated / total)
    efficiency = []
    for results in auction_results.values():
        total = len(results['allocated_tasks']) + len(results['unallocated_tasks'])
        eff = len(results['allocated_tasks']) / total if total > 0 else 0
        efficiency.append(eff * 100)  # Convert to percentage
    
    ax3.bar(auction_types, efficiency)
    ax3.set_xlabel('Auction Type')
    ax3.set_ylabel('Allocation Efficiency (%)')
    ax3.grid(True, alpha=0.3)
    
    # Execution time comparison
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Execution Time Comparison')
    
    times = [results.get('execution_time', 0) for results in auction_results.values()]
    
    ax4.bar(auction_types, times)
    ax4.set_xlabel('Auction Type')
    ax4.set_ylabel('Execution Time (s)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


class SimulationVisualizer:
    """
    Visualizer for the task allocation simulation environment.
    
    This class handles rendering of the simulation environment, including
    robots, tasks, allocations, and information panels, using either
    Pygame or Matplotlib for visualization.
    """
    
    def __init__(self, 
                grid_size: int, 
                task_manager: Any, 
                robots: List,
                render_mode: str = "pygame",
                window_size: int = 800):
        """
        Initialize the visualizer.
        
        Args:
            grid_size: Size of the simulation grid
            task_manager: TaskManager instance
            robots: List of robot objects
            render_mode: Visualization mode ("pygame", "matplotlib", or None)
            window_size: Size of the visualization window in pixels
        """
        self.grid_size = grid_size
        self.task_manager = task_manager
        self.robots = robots
        self.render_mode = render_mode
        self.window_size = window_size
        
        # Calculate cell size for visualization
        self.cell_size = window_size // grid_size
        
        # Pygame resources
        self.screen = None
        self.clock = None
        self.font = None
        
        # Matplotlib resources
        self.fig = None
        self.ax = None
        
        # Current state
        self.allocations = {}
        self.selected_robot_id = None
        self.selected_task_id = None
        self.simulation_info = {}
        self.auction_data = {
            'auction_type': 'sequential',
            'payment_rule': 'first_price',
            'current_phase': 'Idle',
            'metrics': {}
        }
        
        # Initialize visualization environment
        if render_mode == "pygame":
            self._init_pygame()
        elif render_mode == "matplotlib":
            self._init_matplotlib()
    
    def _init_pygame(self):
        """Initialize pygame environment."""
        self.screen, self.clock = setup_pygame(self.window_size, self.window_size, "Task Allocation Simulation")
        self.font = pygame.font.Font(None, 22)
    
    def _init_matplotlib(self):
        """Initialize matplotlib environment."""
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('Task Allocation Simulation')
        
        # Add grid lines
        for i in range(self.grid_size + 1):
            self.ax.axhline(i, color='gray', alpha=0.3)
            self.ax.axvline(i, color='gray', alpha=0.3)
    
    def reset(self):
        """Reset the visualizer state."""
        self.allocations = {}
        self.selected_robot_id = None
        self.selected_task_id = None
        self.simulation_info = {}
        self.auction_data = {
            'auction_type': 'sequential',
            'payment_rule': 'first_price',
            'current_phase': 'Idle',
            'metrics': {}
        }
        
        # Reset visualization environment
        if self.render_mode == "pygame" and self.screen:
            self.screen.fill(COLORS['background'])
        elif self.render_mode == "matplotlib" and self.ax:
            self.ax.clear()
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)
            self.ax.set_aspect('equal')
            self.ax.set_title('Task Allocation Simulation')
            
            # Add grid lines
            for i in range(self.grid_size + 1):
                self.ax.axhline(i, color='gray', alpha=0.3)
                self.ax.axvline(i, color='gray', alpha=0.3)
    
    def render(self, current_step: int):
        """
        Render the current state of the simulation.
        
        Args:
            current_step: Current simulation time step
        """
        # Update simulation information
        self.simulation_info = {
            'Step': current_step,
            'Robots': len(self.robots),
            'Available Tasks': len(self.task_manager.available_tasks),
            'Assigned Tasks': len(self.task_manager.assigned_tasks),
            'Completed Tasks': len(self.task_manager.completed_tasks),
            'Failed Tasks': len(self.task_manager.failed_tasks)
        }
        
        # Build allocations dictionary from TaskManager
        self.allocations = {}
        for task_id in self.task_manager.assigned_tasks:
            task = self.task_manager.tasks.get(task_id)
            if task and hasattr(task, 'assigned_to'):
                self.allocations[task_id] = task.assigned_to
        
        # Render based on mode
        if self.render_mode == "pygame":
            self._render_pygame()
        elif self.render_mode == "matplotlib":
            self._render_matplotlib()
    
    def _render_pygame(self):
        """Render the simulation using pygame."""
        if not self.screen:
            return
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Handle mouse click to select robots/tasks
                mx, my = pygame.mouse.get_pos()
                grid_x = mx // self.cell_size
                grid_y = my // self.cell_size
                
                # Check if clicked on a robot
                for robot in self.robots:
                    rx, ry = robot.position
                    if grid_x == rx and grid_y == ry:
                        self.selected_robot_id = robot.id
                        self.selected_task_id = None
                        break
                
                # Check if clicked on a task
                for task_id in self.task_manager.tasks:
                    task = self.task_manager.tasks[task_id]
                    tx, ty = task.position
                    if grid_x == tx and grid_y == ty:
                        self.selected_task_id = task.id
                        self.selected_robot_id = None
                        break
        
        # Clear screen
        self.screen.fill(COLORS['background'])
        
        # Draw grid
        draw_grid(self.screen, self.grid_size, self.cell_size)
        
        # Get tasks for visualization
        tasks = []
        for task_id in self.task_manager.tasks:
            tasks.append(self.task_manager.tasks[task_id])
        
        # Draw tasks
        draw_tasks(self.screen, tasks, self.cell_size, self.selected_task_id)
        
        # Draw robots
        draw_robots(self.screen, self.robots, self.cell_size, self.selected_robot_id)
        
        # Draw allocations
        draw_allocations(self.screen, tasks, self.robots, self.allocations, self.cell_size)
        
        # Draw info panels
        draw_info_panel(self.screen, self.simulation_info, self.font)
        draw_auction_process(self.screen, self.auction_data, self.font)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)
    
    def _render_matplotlib(self):
        """Render the simulation using matplotlib."""
        if not self.ax:
            return
        
        # Clear previous frame
        self.ax.clear()
        
        # Set up axis
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('Task Allocation Simulation')
        
        # Add grid lines
        for i in range(self.grid_size + 1):
            self.ax.axhline(i, color='gray', alpha=0.3)
            self.ax.axvline(i, color='gray', alpha=0.3)
        
        # Get tasks for visualization
        tasks = []
        for task_id in self.task_manager.tasks:
            tasks.append(self.task_manager.tasks[task_id])
        
        # Draw tasks
        for task in tasks:
            x, y = task.position
            
            # Determine marker and color
            marker = 's'  # Square for regular tasks
            if task.requires_coalition:
                marker = 'd'  # Diamond for coalition tasks
            elif hasattr(task, 'bundle_id') and task.bundle_id:
                marker = 'h'  # Hexagon for bundled tasks
            
            # Determine color
            if task.completed:
                color = 'gray'
            elif task.failed:
                color = 'red'
            elif task.assigned:
                color = 'blue'
            else:
                color = 'green'
            
            # Draw task
            self.ax.scatter(x, y, marker=marker, color=color, s=100, alpha=0.7)
            self.ax.text(x, y, str(task.id), ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # Draw bundle connections
        bundle_tasks = defaultdict(list)
        for task in tasks:
            if hasattr(task, 'bundle_id') and task.bundle_id:
                bundle_tasks[task.bundle_id].append(task)
        
        for bundle_id, bundle_tasks_list in bundle_tasks.items():
            if len(bundle_tasks_list) > 1:
                # Draw lines connecting bundled tasks
                x_coords = [task.position[0] for task in bundle_tasks_list]
                y_coords = [task.position[1] for task in bundle_tasks_list]
                self.ax.plot(x_coords, y_coords, 'c--', alpha=0.7)
        
        # Draw robots
        for robot in self.robots:
            x, y = robot.position
            # Round for display purposes
            display_x, display_y = round(x), round(y)
            
            # Determine color based on strategy type
            color = {
                'truthful': 'blue',
                'strategic': 'orange',
                'learning': 'green',
                'cooperative': 'purple'
            }.get(robot.strategy_type, 'gray')
            
            # Draw robot
            self.ax.scatter(display_x, display_y, marker='o', color=color, s=150, alpha=0.7)
            self.ax.text(x, y, str(robot.id), ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        
        # Draw allocations
        for task_id, robot_id in self.allocations.items():
            task = self.task_manager.tasks.get(task_id)
            robot = next((r for r in self.robots if r.id == robot_id), None)
            
            if task and robot:
                tx, ty = task.position
                rx, ry = robot.position
                # Round for display purposes
                display_rx, display_ry = round(rx), round(ry)
                
                # Determine line color based on robot strategy
                color = {
                    'truthful': 'blue',
                    'strategic': 'orange',
                    'learning': 'green',
                    'cooperative': 'purple'
                }.get(robot.strategy_type, 'gray')
                
                self.ax.plot([display_rx, tx], [display_ry, ty], color=color, alpha=0.5, lw=1)
        
        # Add simulation info text
        info_text = "\n".join([f"{k}: {v}" for k, v in self.simulation_info.items()])
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes, va='top', fontsize=9)
        
        # Display plot
        plt.draw()
        plt.pause(0.001)
    
    def update_auction_data(self, auction_type: str, payment_rule: str, 
                         phase: str, metrics: Dict):
        """
        Update auction visualization data.
        
        Args:
            auction_type: Type of auction mechanism
            payment_rule: Payment rule used
            phase: Current auction phase
            metrics: Auction performance metrics
        """
        self.auction_data = {
            'auction_type': auction_type,
            'payment_rule': payment_rule,
            'current_phase': phase,
            'metrics': metrics
        }
    
    def save_screenshot(self, filename: str):
        """
        Save a screenshot of the current visualization.
        
        Args:
            filename: Path to save the screenshot
        """
        if self.render_mode == "pygame" and self.screen:
            pygame.image.save(self.screen, filename)
        elif self.render_mode == "matplotlib" and self.fig:
            self.fig.savefig(filename)
    
    def close(self):
        """Clean up visualization resources."""
        if self.render_mode == "pygame":
            pygame.quit()
        elif self.render_mode == "matplotlib" and self.fig:
            plt.close(self.fig)


# Example usage if run directly
if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path to import classes
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Importing here to avoid circular imports
    from task_manager import Task, TaskManager
    from bidding_robot import BiddingRobot
    
    # Create synthetic data for demonstration
    # Create task manager
    task_manager = TaskManager(
        environment_size=(20, 20),
        task_complexity_range=(1.0, 3.0)
    )
    
    # Generate tasks
    tasks = task_manager.generate_task_batch(10, include_complex=True)
    
    # Create robots
    robots = [
        BiddingRobot(0, {"speed": 1.5, "lift_capacity": 2.0}, (5, 5), "truthful"),
        BiddingRobot(1, {"speed": 1.0, "precision": 1.8}, (10, 10), "strategic"),
        BiddingRobot(2, {"lift_capacity": 1.2, "sensor_range": 2.0}, (15, 15), "cooperative")
    ]
    
    # Create example allocations
    allocations = {
        tasks[0].id: robots[0].id,
        tasks[1].id: robots[0].id,
        tasks[2].id: robots[1].id,
        tasks[3].id: robots[2].id
    }
    
    # Mark some tasks as completed or failed
    tasks[0].assigned = True
    tasks[1].assigned = True
    tasks[2].assigned = True
    tasks[3].assigned = True
    tasks[4].completed = True
    tasks[5].failed = True
    
    # Test the visualization function
    fig = visualize_allocation_results(tasks, robots, allocations)
    plt.savefig("allocation_visualization_example.png")
    plt.close(fig)
    
    # Test animation creation
    anim = create_animation(robots, tasks, [], allocations, grid_size=20)
    anim.save("allocation_animation_example.mp4", writer='ffmpeg')
    
    print("Visualization examples created successfully.")
    # Note: For the pygame visualizations, we need a running pygame instance with an event loop