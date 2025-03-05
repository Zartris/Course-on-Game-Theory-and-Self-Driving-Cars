#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization utilities for the Roundabout Traffic Management simulation.

This module provides functions for drawing the roundabout, vehicles,
and various information panels using Pygame.
"""

import pygame
import math
import numpy as np

def draw_roundabout(surface, center_x, center_y, radius, lane_width, n_entry_points):
    """
    Draw the roundabout with entry/exit points.
    
    Args:
        surface (pygame.Surface): Surface to draw on
        center_x (int): X-coordinate of roundabout center
        center_y (int): Y-coordinate of roundabout center
        radius (float): Radius of the roundabout
        lane_width (float): Width of the lanes
        n_entry_points (int): Number of entry/exit points
    """
    # Draw outer circle
    pygame.draw.circle(
        surface,
        (100, 100, 100),  # Dark gray
        (center_x, center_y),
        int(radius + lane_width),
        3  # Line width
    )
    
    # Draw inner circle
    pygame.draw.circle(
        surface,
        (100, 100, 100),
        (center_x, center_y),
        int(radius - lane_width),
        3
    )
    
    # Draw entry/exit points
    for i in range(n_entry_points):
        angle = 2 * math.pi * i / n_entry_points
        
        # Calculate entry point coordinates
        entry_x = center_x + (radius + 1.5 * lane_width) * math.cos(angle)
        entry_y = center_y + (radius + 1.5 * lane_width) * math.sin(angle)
        
        # Draw entry road
        road_start_x = center_x + (radius + 2.5 * lane_width) * math.cos(angle)
        road_start_y = center_y + (radius + 2.5 * lane_width) * math.sin(angle)
        
        pygame.draw.line(
            surface,
            (100, 100, 100),
            (int(entry_x - lane_width * math.sin(angle)),
             int(entry_y + lane_width * math.cos(angle))),
            (int(road_start_x - lane_width * math.sin(angle)),
             int(road_start_y + lane_width * math.cos(angle))),
            3
        )
        
        pygame.draw.line(
            surface,
            (100, 100, 100),
            (int(entry_x + lane_width * math.sin(angle)),
             int(entry_y - lane_width * math.cos(angle))),
            (int(road_start_x + lane_width * math.sin(angle)),
             int(road_start_y - lane_width * math.cos(angle))),
            3
        )

def draw_vehicles(surface, vehicles, center_x, center_y):
    """
    Draw all vehicles in the simulation.
    
    Args:
        surface (pygame.Surface): Surface to draw on
        vehicles (list): List of Vehicle objects to draw
        center_x (int): X-coordinate of roundabout center
        center_y (int): Y-coordinate of roundabout center
    """
    # Color mapping for different vehicle types
    color_map = {
        'aggressive': (255, 0, 0),      # Red
        'conservative': (0, 255, 0),    # Green
        'cooperative': (0, 0, 255),     # Blue
        'autonomous': (255, 255, 0)     # Yellow
    }
    
    for vehicle in vehicles:
        # Skip if vehicle has no position
        if not hasattr(vehicle, 'position') or vehicle.position is None:
            continue
        
        # Get vehicle color
        color = color_map.get(vehicle.vehicle_type, (128, 128, 128))
        
        # Draw vehicle body
        pygame.draw.circle(
            surface,
            color,
            (int(vehicle.position[0]), int(vehicle.position[1])),
            int(vehicle.size)
        )
        
        # Draw direction indicator (a line showing vehicle's heading)
        end_x = vehicle.position[0] + vehicle.size * math.cos(vehicle.angle)
        end_y = vehicle.position[1] + vehicle.size * math.sin(vehicle.angle)
        
        pygame.draw.line(
            surface,
            (0, 0, 0),
            (int(vehicle.position[0]), int(vehicle.position[1])),
            (int(end_x), int(end_y)),
            2
        )
        
        # Draw vehicle ID
        font = pygame.font.Font(None, 20)
        id_text = font.render(str(vehicle.id), True, (0, 0, 0))
        text_rect = id_text.get_rect(center=(
            int(vehicle.position[0]),
            int(vehicle.position[1])
        ))
        surface.blit(id_text, text_rect)
        
        # Draw collision indicator if vehicle has collided
        if vehicle.collided:
            pygame.draw.circle(
                surface,
                (255, 0, 0),  # Red
                (int(vehicle.position[0]), int(vehicle.position[1])),
                int(vehicle.size * 1.2),
                2  # Line width
            )

def draw_info_panel(surface, font, info, vehicles):
    """
    Draw information panel with simulation statistics.
    
    Args:
        surface (pygame.Surface): Surface to draw on
        font (pygame.font.Font): Font to use for text
        info (dict): Dictionary containing simulation information
        vehicles (list): List of Vehicle objects
    """
    # Background for info panel
    panel_rect = pygame.Rect(10, 10, 250, 200)
    pygame.draw.rect(surface, (240, 240, 240), panel_rect)
    pygame.draw.rect(surface, (100, 100, 100), panel_rect, 2)
    
    # Display simulation statistics
    y_offset = 20
    line_height = 25
    
    text_items = [
        f"Vehicles: {info['vehicle_count']}",
        f"Completed: {info['vehicles_completed']}",
        f"Collisions: {info['collisions']}",
        f"Avg Wait: {info['mean_wait_time']:.1f}",
        f"Avg Travel: {info['mean_travel_time']:.1f}"
    ]
    
    for text in text_items:
        text_surface = font.render(text, True, (0, 0, 0))
        surface.blit(text_surface, (20, y_offset))
        y_offset += line_height
    
    # Display vehicle type counts
    y_offset += 10
    type_counts = {
        'aggressive': 0,
        'conservative': 0,
        'cooperative': 0,
        'autonomous': 0
    }
    
    for vehicle in vehicles:
        if vehicle.vehicle_type in type_counts:
            type_counts[vehicle.vehicle_type] += 1
    
    text_surface = font.render("Vehicle Types:", True, (0, 0, 0))
    surface.blit(text_surface, (20, y_offset))
    y_offset += line_height
    
    for v_type, count in type_counts.items():
        text_surface = font.render(f"  {v_type}: {count}", True, (0, 0, 0))
        surface.blit(text_surface, (20, y_offset))
        y_offset += line_height