import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class Package:
    """Represents a package with its properties."""
    id: int
    location: Tuple[float, float]  # (x, y) coordinates
    priority: bool  # True if high priority
    weight: float
    delivery_time: float = 0  # Actual delivery time after routing
    assigned_vehicle: int = None  # Vehicle ID currently assigned to deliver this package


@dataclass
class Vehicle:
    """Represents a delivery vehicle with its constraints."""
    id: int
    max_capacity: float
    current_location: Tuple[float, float]
    packages: List[Package] = None
    total_distance: float = 0
    route_duration: float = 0


class DeliveryGene:
    """Represents a single solution in the genetic algorithm."""

    def __init__(self, vehicles: List[Vehicle], packages: List[Package]):
        self.vehicles = vehicles
        self.packages = packages
        self.fitness_score = float('inf')
        # Initialize random valid solution
        self.initialize_random_solution()

    def initialize_random_solution(self):
        """Creates a random valid initial solution."""
        unassigned_packages = self.packages.copy()
        # Sort packages so high priority ones are assigned first
        unassigned_packages.sort(key=lambda x: (not x.priority, random.random()))

        for package in unassigned_packages:
            # Find viable vehicles that can handle this package
            viable_vehicles = [
                v for v in self.vehicles
                if self.can_vehicle_handle_package(v, package)
            ]

            if viable_vehicles:
                # Choose random viable vehicle
                chosen_vehicle = random.choice(viable_vehicles)
                if not chosen_vehicle.packages:
                    chosen_vehicle.packages = []
                chosen_vehicle.packages.append(package)
                package.assigned_vehicle = chosen_vehicle.id

    def can_vehicle_handle_package(self, vehicle: Vehicle, package: Package) -> bool:
        """Checks if vehicle can handle additional package within constraints."""
        if not vehicle.packages:
            return True

        current_weight = sum(p.weight for p in vehicle.packages)
        if current_weight + package.weight > vehicle.max_capacity:
            return False

        # Calculate estimated delivery time for high priority package
        if package.priority:
            estimated_time = self.calculate_route_time(
                vehicle.current_location,
                [p.location for p in vehicle.packages] + [package.location]
            )
            if estimated_time > 2 * 60:  # 2 hours in minutes
                return False

        return True

    def calculate_route_time(self, start: Tuple[float, float],
                             locations: List[Tuple[float, float]]) -> float:
        """Calculates total route time including traffic delays."""
        total_time = 0
        current = start

        for next_loc in locations:
            distance = np.sqrt(
                (next_loc[1] - current[1]) ** 2 +
                (next_loc[0] - current[0]) ** 2
            )
            # Simple traffic multiplier between 1.0 and 2.0
            traffic_multiplier = 1.0 + random.random()
            travel_time = distance * traffic_multiplier
            total_time += travel_time
            current = next_loc

        return total_time

    def calculate_fitness(self, w1: float = 1.0, w2: float = 1.0) -> float:
        """Calculates fitness score based on given formula."""
        total_distance = 0
        delay_penalty = 0
        capacity_penalty = 0

        for vehicle in self.vehicles:
            if not vehicle.packages:
                continue

            # Calculate route metrics
            route_locations = [p.location for p in vehicle.packages]
            route_time = self.calculate_route_time(
                vehicle.current_location,
                route_locations
            )

            # Add penalties for constraint violations
            if route_time > 8 * 60:  # 8 hours in minutes
                delay_penalty += route_time - (8 * 60)

            current_weight = sum(p.weight for p in vehicle.packages)
            if current_weight > vehicle.max_capacity:
                capacity_penalty += current_weight - vehicle.max_capacity

            # Calculate total distance
            current = vehicle.current_location
            for location in route_locations:
                distance = np.sqrt(
                    (location[1] - current[1]) ** 2 +
                    (location[0] - current[0]) ** 2
                )
                total_distance += distance
                current = location

        self.fitness_score = (
                total_distance +
                w1 * delay_penalty +
                w2 * capacity_penalty
        )
        return self.fitness_score

