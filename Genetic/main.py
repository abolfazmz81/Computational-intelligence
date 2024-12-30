import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
import random
from datetime import datetime, timedelta


@dataclass
class Package:
    """Represents a package with its properties."""
    id: int
    location: Tuple[int, int]  # (x, y) coordinates
    priority: bool  # True if high priority
    weight: float
    delivery_time: float = 0  # Actual delivery time after routing
    assigned_vehicle: int = None  # Vehicle ID currently assigned to deliver this package


@dataclass
class Vehicle:
    """Represents a delivery vehicle with its constraints."""
    id: int
    max_capacity: float
    current_location: Tuple[int, int]
    packages: List[Package] = None
    total_distance: float = 0
    route_duration: float = 0


class DistanceMatrix:
    """Manages pre-calculated distances between all grid points."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        # Calculate total number of points in the grid
        self.total_points = grid_size * grid_size
        # Create the distance matrix
        self.matrix = self._initialize_distances()

    def _initialize_distances(self) -> np.ndarray:
        """Creates a matrix containing distances between all grid points using the specified formula."""
        matrix = np.zeros((self.total_points, self.total_points))

        # Calculate distances between all pairs of points
        for i in range(self.total_points):
            # Convert linear index to grid coordinates
            x1, y1 = i // self.grid_size, i % self.grid_size

            for j in range(i + 1, self.total_points):
                # Convert linear index to grid coordinates
                x2, y2 = j // self.grid_size, j % self.grid_size

                # Calculate distance using the formula √((y₂-y₁)² + (x₂-x₁)²)
                distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

                # Distance matrix is symmetric
                matrix[i][j] = distance
                matrix[j][i] = distance

        return matrix

    def get_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Retrieves the pre-calculated distance between two grid points."""
        # Convert grid coordinates to linear indices
        idx1 = point1[0] * self.grid_size + point1[1]
        idx2 = point2[0] * self.grid_size + point2[1]
        return self.matrix[idx1][idx2]


class TrafficSystem:
    """Manages dynamic traffic coefficients that change based on time and location."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        # Define rush hours (8 AM and 5 PM)
        self.rush_hours = [8, 17]
        # Create hourly traffic patterns
        self.hourly_patterns = self._create_hourly_patterns()
        # Create congestion zones
        self.congestion_zones = self._create_congestion_zones()

    def _create_hourly_patterns(self) -> Dict[int, float]:
        """Creates traffic multipliers for each hour of the day."""
        patterns = {}
        for hour in range(24):
            if hour in self.rush_hours:
                # Rush hour traffic (1.5-2.0x)
                patterns[hour] = 1.5 + random.random() * 0.5
            elif any(abs(hour - rush) <= 1 for rush in self.rush_hours):
                # Near rush hour (1.3-1.7x)
                patterns[hour] = 1.3 + random.random() * 0.4
            else:
                # Normal hours (1.0-1.3x)
                patterns[hour] = 1.0 + random.random() * 0.3
        return patterns

    def _create_congestion_zones(self) -> List[Dict]:
        """Creates areas of consistently higher traffic."""
        zones = []
        # Create a congested city center
        center = self.grid_size // 2

        # Add central business district
        zones.append({
            'center': (center, center),
            'radius': 2,
            'multiplier': 1.5 + random.random() * 0.3
        })

        # Add random congestion zones
        num_zones = max(3, self.grid_size // 3)
        for _ in range(num_zones):
            zones.append({
                'center': (random.randint(0, self.grid_size - 1),
                           random.randint(0, self.grid_size - 1)),
                'radius': 1,
                'multiplier': 1.2 + random.random() * 0.3
            })
        return zones

    def get_traffic_coefficient(self, point1: Tuple[int, int], point2: Tuple[int, int],
                                current_time: datetime) -> float:
        """Calculates the current traffic coefficient between two points."""
        hour = current_time.hour
        base_coefficient = self.hourly_patterns[hour]

        # Check if route passes through congestion zones
        for zone in self.congestion_zones:
            if self._route_crosses_zone(point1, point2, zone):
                base_coefficient *= zone['multiplier']

        # Add random variation (±10%)
        variation = 0.9 + random.random() * 0.2

        return base_coefficient * variation

    def _route_crosses_zone(self, start: Tuple[int, int], end: Tuple[int, int],
                            zone: Dict) -> bool:
        """Determines if a route passes through a congestion zone."""
        zone_center = zone['center']
        # Check if either endpoint is in the zone
        for point in [start, end]:
            distance_to_center = np.sqrt(
                (point[0] - zone_center[0]) ** 2 +
                (point[1] - zone_center[1]) ** 2
            )
            if distance_to_center <= zone['radius']:
                return True
        return False


class DeliveryGene:
    """Represents a single solution in the genetic algorithm."""

    def __init__(self, grid_size: int, vehicles: List[Vehicle], packages: List[Package],
                 distance_matrix: DistanceMatrix, traffic_system: TrafficSystem):
        self.grid_size = grid_size
        self.vehicles = vehicles
        self.packages = packages
        self.distance_matrix = distance_matrix
        self.traffic_system = traffic_system
        self.fitness_score = float('inf')
        self.PRIORITY_TIME_LIMIT = 2 * 60  # 2 hours in minutes
        self.WORKDAY_LIMIT = 8 * 60  # 8 hours in minutes
        self.start_time = datetime.now().replace(hour=8, minute=0)  # Start at 8 AM
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
            if estimated_time > self.PRIORITY_TIME_LIMIT:
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

    def calculate_route_times(self, vehicle: Vehicle, route: List[Tuple[int, int]]) -> List[float]:
        """Calculates delivery times for each stop on a route."""
        current_time = self.start_time
        current_location = vehicle.current_location
        delivery_times = []

        for location in route:
            distance = self.distance_matrix.get_distance(current_location, location)
            traffic_coef = self.traffic_system.get_traffic_coefficient(
                current_location, location, current_time
            )

            travel_time = distance * traffic_coef
            current_time += timedelta(minutes=int(travel_time))
            delivery_times.append((current_time - self.start_time).total_seconds() / 60)
            current_location = location

        return delivery_times

    def calculate_fitness(self, w1: float = 1.0, w2: float = 2.0, w3: float = 3.0) -> float:
        """
        Calculates fitness with priority package time penalties.
        w1: weight for general delay penalty
        w2: weight for capacity penalty
        w3: weight for priority delivery time penalty
        """
        total_distance = 0
        delay_penalty = 0
        capacity_penalty = 0
        priority_time_penalty = 0

        for vehicle in self.vehicles:
            if not vehicle.packages:
                continue

            # Get route locations
            route_locations = [p.location for p in vehicle.packages]

            # Calculate delivery times
            delivery_times = self.calculate_route_times(vehicle, route_locations)

            # Check constraints and update penalties
            for package, delivery_time in zip(vehicle.packages, delivery_times):
                if package.priority and delivery_time > self.PRIORITY_TIME_LIMIT:
                    priority_time_penalty += (delivery_time - self.PRIORITY_TIME_LIMIT)

                package.delivery_time = delivery_time

                if delivery_time > self.WORKDAY_LIMIT:
                    delay_penalty += delivery_time - self.WORKDAY_LIMIT

            # Check capacity constraint
            current_weight = sum(p.weight for p in vehicle.packages)
            if current_weight > vehicle.max_capacity:
                capacity_penalty += current_weight - vehicle.max_capacity

            # Calculate total distance
            current = vehicle.current_location
            for location in route_locations:
                distance = self.distance_matrix.get_distance(current, location)
                total_distance += distance
                current = location

        # Calculate final fitness score using all penalties
        self.fitness_score = (
                total_distance +
                w1 * delay_penalty +
                w2 * capacity_penalty +
                w3 * priority_time_penalty  # Higher weight for priority package delays
        )
        return self.fitness_score

