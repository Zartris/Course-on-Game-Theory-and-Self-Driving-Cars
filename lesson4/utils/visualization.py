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

def draw_city_grid(surface, grid_size, intersection_size):
    """Draw the city grid with emphasis on the 4-way intersection."""
    # Get surface dimensions
    width, height = surface.get_width(), surface.get_height()
    
    # Scaling factors
    scale_x = width / grid_size
    scale_y = height / grid_size
    
    # Fill background
    surface.fill((200, 200, 200))  # Light gray for non-road areas
    
    # Calculate intersection center and size
    center_x = grid_size // 2
    center_y = grid_size // 2
    road_width = intersection_size * 2  # Width of roads is twice the intersection size
    
    # Draw the four roads
    # Horizontal road
    pygame.draw.rect(
        surface,
        (100, 100, 100),  # Dark gray for roads
        (0, (center_y - road_width/2) * scale_y, 
         width, road_width * scale_y)
    )
    
    # Vertical road
    pygame.draw.rect(
        surface,
        (100, 100, 100),  # Dark gray for roads
        ((center_x - road_width/2) * scale_x, 0,
         road_width * scale_x, height)
    )
    
    # Draw intersection box
    pygame.draw.rect(
        surface,
        (120, 120, 120),  # Slightly darker for intersection
        ((center_x - road_width/2) * scale_x,
         (center_y - road_width/2) * scale_y,
         road_width * scale_x,
         road_width * scale_y)
    )
    
    # Draw road markings
    line_color = (255, 255, 255)  # White for road markings
    dash_length = scale_x * 0.5  # Scale dash length with screen size
    gap_length = scale_x * 0.5
    
    # Horizontal road markings - center line
    y_center = center_y * scale_y
    x_start = 0
    while x_start < width:
        if x_start < (center_x - road_width/2) * scale_x or x_start > (center_x + road_width/2) * scale_x:
            pygame.draw.line(
                surface,
                line_color,
                (x_start, y_center),
                (min(x_start + dash_length, width), y_center),
                2
            )
        x_start += dash_length + gap_length
    
    # Vertical road markings - center line
    x_center = center_x * scale_x
    y_start = 0
    while y_start < height:
        if y_start < (center_y - road_width/2) * scale_y or y_start > (center_y + road_width/2) * scale_y:
            pygame.draw.line(
                surface,
                line_color,
                (x_center, y_start),
                (x_center, min(y_start + dash_length, height)),
                2
            )
        y_start += dash_length + gap_length
    
    # Draw stop lines at intersection
    line_thickness = max(2, int(scale_x * 0.1))  # Scale line thickness
    
    # North stop line
    pygame.draw.line(
        surface,
        (255, 255, 255),
        ((center_x - road_width/2) * scale_x, (center_y - road_width/2) * scale_y),
        ((center_x + road_width/2) * scale_x, (center_y - road_width/2) * scale_y),
        line_thickness
    )
    
    # South stop line
    pygame.draw.line(
        surface,
        (255, 255, 255),
        ((center_x - road_width/2) * scale_x, (center_y + road_width/2) * scale_y),
        ((center_x + road_width/2) * scale_x, (center_y + road_width/2) * scale_y),
        line_thickness
    )
    
    # West stop line
    pygame.draw.line(
        surface,
        (255, 255, 255),
        ((center_x - road_width/2) * scale_x, (center_y - road_width/2) * scale_y),
        ((center_x - road_width/2) * scale_x, (center_y + road_width/2) * scale_y),
        line_thickness
    )
    
    # East stop line
    pygame.draw.line(
        surface,
        (255, 255, 255),
        ((center_x + road_width/2) * scale_x, (center_y - road_width/2) * scale_y),
        ((center_x + road_width/2) * scale_x, (center_y + road_width/2) * scale_y),
        line_thickness
    )
    
    # Draw direction arrows on roads
    arrow_color = (255, 255, 255)
    arrow_length = scale_x * 1.0  # Scale arrow size with screen
    arrow_width = scale_x * 0.5
    
    # Helper function to draw arrow
    def draw_arrow(surface, start_pos, direction):
        end_pos = (
            start_pos[0] + direction[0] * arrow_length,
            start_pos[1] + direction[1] * arrow_length
        )
        
        # Draw arrow shaft
        pygame.draw.line(surface, arrow_color, start_pos, end_pos, max(2, int(scale_x * 0.1)))
        
        # Calculate arrow head points
        angle = math.pi / 6  # 30 degrees
        direction_norm = math.sqrt(direction[0]**2 + direction[1]**2)
        dx = direction[0] / direction_norm
        dy = direction[1] / direction_norm
        
        right_x = end_pos[0] - arrow_width * (dx * math.cos(angle) + dy * math.sin(angle))
        right_y = end_pos[1] - arrow_width * (-dx * math.sin(angle) + dy * math.cos(angle))
        
        left_x = end_pos[0] - arrow_width * (dx * math.cos(-angle) + dy * math.sin(-angle))
        left_y = end_pos[1] - arrow_width * (-dx * math.sin(-angle) + dy * math.cos(-angle))
        
        # Draw arrow head
        pygame.draw.polygon(surface, arrow_color, [end_pos, (right_x, right_y), (left_x, left_y)])
    
    # Draw multiple arrows for each direction
    arrow_spacing = road_width * scale_x * 0.8
    
    # Left to right arrows
    for x in range(3):
        draw_arrow(surface, 
                ((center_x - 2*road_width + x*road_width) * scale_x, 
                (center_y - road_width/4) * scale_y),
                (1, 0))
    
    # Right to left arrows
    for x in range(3):
        draw_arrow(surface,
                ((center_x + 2*road_width - x*road_width) * scale_x,
                (center_y + road_width/4) * scale_y),
                (-1, 0))
    
    # Top to bottom arrows
    for y in range(3):
        draw_arrow(surface,
                ((center_x - road_width/4) * scale_x,
                (center_y - 2*road_width + y*road_width) * scale_y),
                (0, 1))
    
    # Bottom to top arrows
    for y in range(3):
        draw_arrow(surface,
                ((center_x + road_width/4) * scale_x,
                (center_y + 2*road_width - y*road_width) * scale_y),
                (0, -1))

def draw_vehicles(screen, vehicles):
    """Draw vehicles and their paths on the screen."""
    # Get screen dimensions
    width = screen.get_width() - 200  # Adjust for info panel
    height = screen.get_height()
    
    for vehicle in vehicles:
        # Draw vehicle position
        pos_x = vehicle.position[0] * (width / 20)
        pos_y = vehicle.position[1] * (height / 20)
        
        # Draw goal position
        goal_x = vehicle.final_goal[0] * (width / 20)
        goal_y = vehicle.final_goal[1] * (height / 20)
        
        # Set color based on vehicle type
        if vehicle.vehicle_type == 'emergency':
            color = (255, 0, 0)  # Red for emergency
        elif vehicle.vehicle_type == 'premium':
            color = (0, 255, 0)  # Green for premium
        else:
            color = (0, 0, 255)  # Blue for standard
        
        # Draw vehicle as circle
        pygame.draw.circle(screen, color, (int(pos_x), int(pos_y)), 8)
        
        # Draw emergency vehicle indicator
        if vehicle.vehicle_type == 'emergency':
            pygame.draw.circle(screen, color, (int(pos_x), int(pos_y)), 12, 2)
        
        # Draw goal as small circle
        pygame.draw.circle(screen, color, (int(goal_x), int(goal_y)), 3, 1)
        
        # Draw line to goal
        pygame.draw.line(screen, color, (pos_x, pos_y), (goal_x, goal_y), 1)
        
        # If waiting at intersection, draw indicator
        if hasattr(vehicle, 'waiting_at_intersection') and vehicle.waiting_at_intersection:
            pygame.draw.circle(screen, (255, 255, 0), (int(pos_x), int(pos_y)), 10, 1)
        
        # If giving way to emergency vehicle, draw indicator
        if hasattr(vehicle, 'give_way_to_emergency') and vehicle.give_way_to_emergency:
            pygame.draw.circle(screen, (255, 165, 0), (int(pos_x), int(pos_y)), 12, 1)

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