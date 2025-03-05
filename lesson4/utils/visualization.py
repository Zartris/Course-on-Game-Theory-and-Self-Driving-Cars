#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for the fleet coordination simulation.

This module provides functions for rendering the city grid, vehicles,
belief states, and information flow in the simulation.
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import math

def draw_city_grid(surface, grid_size, block_size):
    """
    Draw the city grid with blocks and roads.
    
    Args:
        surface: Pygame surface to draw on
        grid_size (int): Size of the grid
        block_size (int): Size of city blocks
    """
    # Get surface dimensions
    width, height = surface.get_width(), surface.get_height()
    
    # Scaling factors
    scale_x = width / grid_size
    scale_y = height / grid_size
    
    # Draw blocks
    for i in range(0, grid_size, block_size):
        for j in range(0, grid_size, block_size):
            # Skip road positions
            if i % block_size == 0 or j % block_size == 0:
                continue
                
            # Calculate block coordinates
            block_x = i * scale_x
            block_y = j * scale_y
            block_width = min(block_size, grid_size - i) * scale_x
            block_height = min(block_size, grid_size - j) * scale_y
            
            # Draw block as light gray rectangle
            pygame.draw.rect(
                surface,
                (200, 200, 200),
                (block_x, block_y, block_width, block_height)
            )
    
    # Draw grid lines for roads
    for i in range(0, grid_size + 1):
        # Horizontal roads
        pygame.draw.line(
            surface,
            (100, 100, 100),
            (0, i * scale_y),
            (width, i * scale_y),
            1 if i % block_size != 0 else 2
        )
        
        # Vertical roads
        pygame.draw.line(
            surface,
            (100, 100, 100),
            (i * scale_x, 0),
            (i * scale_x, height),
            1 if i % block_size != 0 else 2
        )
    
    # Highlight intersections
    for i in range(block_size, grid_size, block_size):
        for j in range(block_size, grid_size, block_size):
            pygame.draw.circle(
                surface,
                (150, 150, 150),
                (i * scale_x, j * scale_y),
                5
            )

def draw_vehicles(surface, vehicles):
    """
    Draw all vehicles on the surface.
    
    Args:
        surface: Pygame surface to draw on
        vehicles (list): List of BayesianVehicle objects
    """
    # Get surface dimensions
    width, height = surface.get_width(), surface.get_height()
    
    # Draw each vehicle
    for vehicle in vehicles:
        # Scale position to surface
        pos_x = vehicle.position[0] * (width / 20)
        pos_y = vehicle.position[1] * (height / 20)
        
        # Choose color based on vehicle type
        if vehicle.vehicle_type == 'standard':
            color = (0, 0, 255)  # Blue
        elif vehicle.vehicle_type == 'premium':
            color = (0, 255, 0)  # Green
        elif vehicle.vehicle_type == 'emergency':
            color = (255, 0, 0)  # Red
        else:
            color = (255, 255, 255)  # White
        
        # Draw vehicle as a circle - larger size
        pygame.draw.circle(surface, color, (int(pos_x), int(pos_y)), 12)
        
        # Draw direction indicator
        if vehicle.velocity[0] != 0 or vehicle.velocity[1] != 0:
            direction = np.array(vehicle.velocity)
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 0.001:  # Avoid division by zero
                direction = direction / direction_norm
                endpoint = (
                    int(pos_x + direction[0] * 20),  # Longer direction indicator
                    int(pos_y + direction[1] * 20)
                )
                pygame.draw.line(surface, color, (int(pos_x), int(pos_y)), endpoint, 3)  # Thicker line
        
        # Draw vehicle ID - larger font
        font = pygame.font.SysFont('Arial', 14, bold=True)  # Larger, bold font
        text_surface = font.render(str(vehicle.id), True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(int(pos_x), int(pos_y)))
        surface.blit(text_surface, text_rect)
        
        # Draw goal indicator - larger
        goal_x = vehicle.goal[0] * (width / 20)
        goal_y = vehicle.goal[1] * (height / 20)
        pygame.draw.circle(surface, color, (int(goal_x), int(goal_y)), 5, 2)  # Larger goal indicator
        
        # Draw line to goal - thicker
        pygame.draw.line(surface, color, (int(pos_x), int(pos_y)), (int(goal_x), int(goal_y)), 2)  # Thicker goal line
        
        # Highlight emergency vehicles on active duty - larger highlight
        if vehicle.vehicle_type == 'emergency' and getattr(vehicle, 'emergency_active', False):
            pygame.draw.circle(surface, (255, 0, 0), (int(pos_x), int(pos_y)), 18, 3)  # Larger highlight
            
            # Draw pulsating effect - larger
            pygame.draw.circle(
                surface,
                (255, 100, 100),
                (int(pos_x), int(pos_y)),
                22,  # Larger pulsating effect
                2    # Thicker line
            )

def draw_belief_states(surface, vehicles):
    """
    Visualize the belief states of vehicles about others' types.
    
    Args:
        surface: Pygame surface to draw on
        vehicles (list): List of BayesianVehicle objects
    """
    # Get surface dimensions
    width, height = surface.get_width(), surface.get_height()
    
    # Draw belief visualizations for each vehicle
    for vehicle in vehicles:
        # Skip if no beliefs are available
        if not hasattr(vehicle, 'beliefs') or not vehicle.beliefs:
            continue
            
        # Scale position to surface
        pos_x = vehicle.position[0] * (width / 20)
        pos_y = vehicle.position[1] * (height / 20)
        
        # Create belief visualization for each belief about another vehicle
        for other_id, type_beliefs in vehicle.beliefs.items():
            # Find the other vehicle
            other_vehicle = next((v for v in vehicles if v.id == other_id), None)
            if not other_vehicle:
                continue
                
            # Scale other position to surface
            other_x = other_vehicle.position[0] * (width / 20)
            other_y = other_vehicle.position[1] * (height / 20)
            
            # Calculate midpoint with slight offset to avoid overlapping with other belief displays
            direction = np.array([other_x - pos_x, other_y - pos_y])
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 0.001:  # Avoid division by zero
                continue
                
            # Normalize and apply offset (30% from vehicle to midpoint)
            direction = direction / direction_norm
            mid_x = pos_x + 0.3 * (other_x - pos_x)
            mid_y = pos_y + 0.3 * (other_y - pos_y)
            
            # Draw connection line (more visible)
            pygame.draw.line(
                surface,
                (120, 120, 120), 
                (int(pos_x), int(pos_y)),
                (int(mid_x), int(mid_y)),
                2  # Thicker line
            )
            
            # Draw belief pie chart at midpoint - larger
            radius = 18  # Larger radius for better visibility
            start_angle = 0
            
            # Sort beliefs by type
            sorted_types = sorted(type_beliefs.items())
            
            # Draw background circle for the belief pie
            pygame.draw.circle(
                surface,
                (240, 240, 240, 220),  # Light gray with more opacity
                (int(mid_x), int(mid_y)),
                radius
            )
            
            # Draw belief segments
            for type_name, probability in sorted_types:
                # Calculate angle based on probability
                angle = 2 * math.pi * probability
                
                # Choose color based on type - more saturated colors
                if type_name == 'standard':
                    color = (0, 0, 255, 200)  # Blue with more opacity
                elif type_name == 'premium':
                    color = (0, 255, 0, 200)  # Green with more opacity
                elif type_name == 'emergency':
                    color = (255, 0, 0, 200)  # Red with more opacity
                else:
                    color = (255, 255, 255, 200)  # White with more opacity
                
                # Draw arc
                if probability > 0.01:  # Only draw if probability is significant
                    points = [
                        (mid_x, mid_y)
                    ]
                    for i in range(21):
                        theta = start_angle + (i / 20.0) * angle
                        x = mid_x + radius * math.cos(theta)
                        y = mid_y + radius * math.sin(theta)
                        points.append((x, y))
                    
                    if len(points) > 2:
                        pygame.draw.polygon(surface, color, points)
                
                # Update start angle
                start_angle += angle
            
            # Draw circle outline
            pygame.draw.circle(
                surface,
                (0, 0, 0),
                (int(mid_x), int(mid_y)),
                radius,
                2  # Thicker outline
            )
            
            # Draw the ID of the vehicle that these beliefs are about - larger font
            font = pygame.font.SysFont('Arial', 12, bold=True)  # Larger, bold font
            text = f"Belief about V{other_id}"  # More descriptive label
            text_surface = font.render(text, True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=(int(mid_x), int(mid_y) - radius - 10))
            surface.blit(text_surface, text_rect)

def draw_info_panel(surface, metrics, emergency_active):
    """
    Draw information panel with metrics and emergency status.
    
    Args:
        surface: Pygame surface to draw on
        metrics (dict): Dictionary of metrics
        emergency_active (bool): Whether emergency scenario is active
    """
    # Get surface dimensions
    width, height = surface.get_width(), surface.get_height()
    
    # Draw panel background
    panel_rect = pygame.Rect(width - 200, 0, 200, height)
    pygame.draw.rect(surface, (240, 240, 240), panel_rect)
    pygame.draw.line(surface, (0, 0, 0), (width - 200, 0), (width - 200, height), 2)
    
    # Create font
    font = pygame.font.SysFont('Arial', 14)
    title_font = pygame.font.SysFont('Arial', 16, bold=True)
    
    # Draw title
    title_surface = title_font.render("Simulation Metrics", True, (0, 0, 0))
    surface.blit(title_surface, (width - 190, 10))
    
    # Draw metrics
    y_offset = 40
    for name, value in metrics.items():
        if name == "emergency_response_time" and value:
            # Format as average response time
            avg_time = sum(value) / len(value) if value else 0
            text = f"{name.replace('_', ' ').title()}: {avg_time:.1f}"
        elif isinstance(value, (int, float)):
            text = f"{name.replace('_', ' ').title()}: {value}"
        else:
            text = f"{name.replace('_', ' ').title()}: {len(value) if value else 0}"
        
        text_surface = font.render(text, True, (0, 0, 0))
        surface.blit(text_surface, (width - 190, y_offset))
        y_offset += 25
    
    # Draw emergency status
    emergency_text = "EMERGENCY ACTIVE" if emergency_active else "No emergency"
    emergency_color = (255, 0, 0) if emergency_active else (0, 100, 0)
    
    # Draw with background if active
    if emergency_active:
        pygame.draw.rect(
            surface,
            (255, 200, 200),
            (width - 190, y_offset, 180, 25)
        )
    
    emergency_surface = font.render(emergency_text, True, emergency_color)
    surface.blit(emergency_surface, (width - 190, y_offset))

def create_belief_heatmap(vehicles, grid_size):
    """
    Create a heatmap visualization of belief states.
    
    Args:
        vehicles (list): List of BayesianVehicle objects
        grid_size (int): Size of the grid
        
    Returns:
        pygame.Surface: Surface with the belief heatmap
    """
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create empty grid for each vehicle type
    standard_grid = np.zeros((grid_size, grid_size))
    premium_grid = np.zeros((grid_size, grid_size))
    emergency_grid = np.zeros((grid_size, grid_size))
    
    # Collect beliefs about vehicle types
    for vehicle in vehicles:
        if not hasattr(vehicle, 'beliefs') or not vehicle.beliefs:
            continue
            
        for other_id, beliefs in vehicle.beliefs.items():
            # Find the other vehicle
            other_vehicle = next((v for v in vehicles if v.id == other_id), None)
            if not other_vehicle:
                continue
            
            # Extract position (rounded to integer grid coordinates)
            pos_x = min(int(other_vehicle.position[0]), grid_size - 1)
            pos_y = min(int(other_vehicle.position[1]), grid_size - 1)
            
            # Add belief probabilities to grids
            standard_grid[pos_y, pos_x] += beliefs.get('standard', 0.0)
            premium_grid[pos_y, pos_x] += beliefs.get('premium', 0.0)
            emergency_grid[pos_y, pos_x] += beliefs.get('emergency', 0.0)
    
    # Normalize grids
    max_belief = max(
        np.max(standard_grid),
        np.max(premium_grid),
        np.max(emergency_grid)
    )
    if max_belief > 0:
        standard_grid /= max_belief
        premium_grid /= max_belief
        emergency_grid /= max_belief
    
    # Create RGB image
    rgb_grid = np.zeros((grid_size, grid_size, 3))
    rgb_grid[:,:,0] = emergency_grid  # Red channel for emergency
    rgb_grid[:,:,1] = premium_grid    # Green channel for premium
    rgb_grid[:,:,2] = standard_grid   # Blue channel for standard
    
    # Plot heatmap
    ax.imshow(rgb_grid, interpolation='bilinear')
    ax.set_title("Vehicle Type Belief Distribution")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0,0,1), markersize=10, label='Standard'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0,1,0), markersize=10, label='Premium'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(1,0,0), markersize=10, label='Emergency')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Convert to pygame surface
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Updated method to get image data from canvas
    width, height = canvas.get_width_height()
    buffer_rgba = np.frombuffer(canvas.buffer_rgba(), np.uint8)
    buffer_rgba = buffer_rgba.reshape((height, width, 4))
    
    # Convert RGBA to RGB
    buffer_rgb = buffer_rgba[:,:,:3]
    
    # Create pygame surface
    surf = pygame.image.frombuffer(buffer_rgb.tobytes(), (width, height), "RGB")
    
    # Close figure to avoid memory leak
    plt.close(fig)
    
    return surf

def visualize_information_flow(surface, vehicles, communication_graph):
    """
    Visualize information flow between vehicles.
    
    Args:
        surface: Pygame surface to draw on
        vehicles (list): List of BayesianVehicle objects
        communication_graph: NetworkX graph of communications
    """
    # Get surface dimensions
    width, height = surface.get_width(), surface.get_height()
    
    # Create mapping of vehicle IDs to positions
    positions = {}
    for vehicle in vehicles:
        pos_x = vehicle.position[0] * (width / 20)
        pos_y = vehicle.position[1] * (height / 20)
        positions[vehicle.id] = (pos_x, pos_y)
    
    # Draw edges representing information flow
    for u, v, data in communication_graph.edges(data=True):
        if u in positions and v in positions:
            start_pos = positions[u]
            end_pos = positions[v]
            
            # Get weight/intensity if available
            weight = data.get('weight', 1.0)
            
            # Higher intensity for stronger connections (lower weight)
            intensity = int(min(255, max(80, 255 * (1.0 - weight / 10.0))))
            
            # Get vehicle types (for coloring)
            u_vehicle = next((veh for veh in vehicles if veh.id == u), None)
            v_vehicle = next((veh for veh in vehicles if veh.id == v), None)
            
            if not u_vehicle or not v_vehicle:
                continue
            
            # Set base color of communication based on sending vehicle type
            if u_vehicle.vehicle_type == 'emergency':
                color = (intensity, 0, 0)  # Red for emergency
            elif u_vehicle.vehicle_type == 'premium':
                color = (0, intensity, 0)  # Green for premium
            else:
                color = (0, 0, intensity)  # Blue for standard
            
            # Calculate a slightly offset line to avoid overlapping with belief visualization
            direction = np.array(end_pos) - np.array(start_pos)
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 0.001:  # Avoid division by zero
                continue
                
            direction = direction / direction_norm
            perp = np.array([-direction[1], direction[0]])
            
            # Offset in the perpendicular direction
            offset_factor = 5
            offset_start = np.array(start_pos) + perp * offset_factor
            offset_end = np.array(end_pos) + perp * offset_factor
            
            # Draw a dotted or dashed line
            dash_length = 5
            gap_length = 3
            dash_count = int(direction_norm / (dash_length + gap_length))
            
            for i in range(dash_count):
                start_frac = i * (dash_length + gap_length) / direction_norm
                end_frac = min(1.0, (i * (dash_length + gap_length) + dash_length) / direction_norm)
                
                dash_start = tuple(map(int, offset_start + direction * start_frac * direction_norm))
                dash_end = tuple(map(int, offset_start + direction * end_frac * direction_norm))
                
                pygame.draw.line(
                    surface,
                    color,
                    dash_start,
                    dash_end,
                    2
                )
            
            # Draw arrowhead
            arrowhead_length = 8
            arrow_angle = 0.5  # ~30 degrees
            
            arrowhead_pos = offset_start + direction * (direction_norm - arrowhead_length)
            
            point1 = arrowhead_pos + direction * arrowhead_length + perp * arrowhead_length * arrow_angle
            point2 = arrowhead_pos + direction * arrowhead_length - perp * arrowhead_length * arrow_angle
            
            pygame.draw.polygon(
                surface,
                color,
                [tuple(map(int, offset_end)), 
                 tuple(map(int, point1)), 
                 tuple(map(int, point2))]
            )
            
            # Add a small info marker showing communication strength
            # Higher reliability = larger marker
            reliability = 1.0 - (weight / 10.0)
            
            # Draw circle at midpoint of communication line
            mid_point = tuple(map(int, offset_start + direction * direction_norm * 0.6))
            
            # Add glow effect for strong communications
            if reliability > 0.6:
                for r in range(5, 2, -1):
                    alpha = int(255 * (reliability - 0.5) / 0.5)
                    glow_color = (*color[:2], min(color[2], 150), alpha)
                    pygame.draw.circle(
                        surface,
                        glow_color,
                        mid_point,
                        r
                    )