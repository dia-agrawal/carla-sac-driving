import math
import heapq
from collections import defaultdict
from queue import PriorityQueue

import carla

# ----------------------------
# Sync + spawn
# ----------------------------
def set_sync(world: carla.World, tm: carla.TrafficManager, fixed_dt: float = 0.05):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = fixed_dt
    settings.max_substeps = 1
    settings.max_substep_delta_time = fixed_dt
    world.apply_settings(settings)
    tm.set_synchronous_mode(True)
    world.tick()

def spawn_ego(world: carla.World, model_filter="vehicle.micro.microlino", spawn_index=0):
    bp = world.get_blueprint_library().filter(model_filter)[0]
    sp = world.get_map().get_spawn_points()[spawn_index]
    veh = world.spawn_actor(bp, sp)
    world.tick()
    return veh

# ----------------------------
# Graph utilities
# ----------------------------
def wp_key(wp: carla.Waypoint):
    """
    Stable-ish key for topology nodes.
    Avoid raw floats; quantize s.
    """
    return (wp.road_id, wp.section_id, wp.lane_id, round(wp.s, 2))

def loc_of_wp(wp: carla.Waypoint):
    return wp.transform.location

def dist_loc(a: carla.Location, b: carla.Location) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

def build_topology_graph(carla_map: carla.Map):
    """
    Returns:
      nodes: key -> representative waypoint
      adj:   key -> list[(neighbor_key, cost)]
    """
    topology = carla_map.get_topology()  # list[(wp_from, wp_to)]
    nodes = {}
    adj = defaultdict(list)

    for wp_from, wp_to in topology:
        k1, k2 = wp_key(wp_from), wp_key(wp_to)
        nodes.setdefault(k1, wp_from)
        nodes.setdefault(k2, wp_to)
        cost = dist_loc(loc_of_wp(wp_from), loc_of_wp(wp_to))
        adj[k1].append((k2, cost))

    return nodes, adj

def snap_to_graph_node(nodes, start_wp: carla.Waypoint):
    """
    Snap start_wp to the closest graph node on the SAME lane (road/section/lane),
    falling back to nearest overall if none match.
    """
    target_lane = (start_wp.road_id, start_wp.section_id, start_wp.lane_id)

    best_k = None
    best_d = float("inf") #positve infinity 

    # first pass: same lane (find the closest waypoint)
    for k, wp in nodes.items(): 
        if (wp.road_id, wp.section_id, wp.lane_id) != target_lane: 
            continue #if not same lane, skip
        d = dist_loc(loc_of_wp(wp), loc_of_wp(start_wp))
        if d < best_d:
            best_d, best_k = d, k

    # fallback: nearest overall
    if best_k is None: #nodes collided (due to rounding?), or no nodes on same lane
        for k, wp in nodes.items():
            d = dist_loc(loc_of_wp(wp), loc_of_wp(start_wp))
            if d < best_d:
                best_d, best_k = d, k

    return best_k

# ----------------------------
# A* search
# ----------------------------
def astar(nodes, adj, start_k, goal_k):
    # """
    # Implement A*.
    # - g[k] = best known cost to reach k
    # - f[k] = g[k] + heuristic(k, goal)
    # - came_from to reconstruct

    # Returns: list of node keys from start -> goal (inclusive)
    # """
    # goal_loc = loc_of_wp(nodes[goal_k])

    # def h(k):
    #     return dist_loc(loc_of_wp(nodes[k]), goal_loc)
    
    # count = 0
    # g_score = {start_k: 0.0} #best known actual cost from start to node n
    # open_set = PriorityQueue() #what to explore next 
    # came_from = {} #to reconstruct path later
    # open_set_hash = {start_k} 
    
    # open_set.put((0, count, start_k))
    
    # while not open_set.empty(): 
    #     current = open_set.get()[2]
    #     open_set_hash.remove(current)
        
    #     if current == goal_k: 
    #         return reconstruct_path(came_from, current)
    #     #key -> list[(neighbor_key, cost)]
    #     for neighbor_key, cost in adj[current]:
    #         temp_g_score = g_score[current] + cost 
    #         if neighbor_key not in g_score: 
    #             g_score[neighbor_key] = float("inf") 
    #         if temp_g_score < g_score[neighbor_key]: 
    #             came_from[neighbor_key] = current 
    #             g_score[neighbor_key] = temp_g_score
    #             f_score[neighbor_key] = temp_g_score + h(neighbor_key)
                
    #             if neighbor_key not in open_set_hash: 
    #                 count += 1
    #                 open_set.put((f_score[neighbor_key], count, neighbor_key))      
    #                 open_set_hash.add(neighbor_key)
                    
    
    
    # # TODO: implement with heapq.
    # # Suggested heap items: (f_score, tie_breaker, node_key)
    # # Track: g_score dict, came_from dict, closed set
    # raise NotImplementedError
    if start_k == goal_k:
        return [start_k]

    goal_loc = loc_of_wp(nodes[goal_k])

    def h(k):
        return dist_loc(loc_of_wp(nodes[k]), goal_loc)

    count = 0
    came_from = {}

    # Best known true cost from start to each node
    g_score = {start_k: 0.0}

    # Priority queue entries are (f_score, tie_breaker, node_key)
    open_set = PriorityQueue()
    open_set.put((h(start_k), count, start_k))

    closed = set()

    while not open_set.empty():
        f_cur, _, current = open_set.get()

        # Skip stale/outdated queue entries (because we allow duplicates)
        best_f_current = g_score.get(current, float("inf")) + h(current)
        if f_cur > best_f_current + 1e-6:
            continue

        if current == goal_k:
            return reconstruct_path(came_from, current)

        if current in closed:
            continue
        closed.add(current)

        # Expand neighbors of current
        for neighbor_key, edge_cost in adj[current]:
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor_key, float("inf")):
                came_from[neighbor_key] = current
                g_score[neighbor_key] = tentative_g

                count += 1
                f_neighbor = tentative_g + h(neighbor_key)
                open_set.put((f_neighbor, count, neighbor_key))

    # No path found
    return []
    

def reconstruct_path(came_from, cur):
    path = [cur]
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

# ----------------------------
# Densify node path into ~2m waypoints (you implement)
# ----------------------------
def densify_path(nodes, node_path_keys, resolution_m=2.0):
    """
    Turn sparse topology nodes into a dense list of carla.Waypoint spaced ~resolution_m.
    Typical strategy:
      for each consecutive (wp_a, wp_b):
        walk forward from wp_a using wp.next(resolution_m) until close to wp_b.
    """
    dense = []
    # TODO: implement
    # Tips:
    current = wp_a
    while dist(current, wp_b) > resolution_m:
      nxt = current.next(resolution_m)
      if not nxt: break
      current = nxt[0]   # usually one forward wp
      dense.append(current)
    dense.append(wp_b) 
    return dense

# ----------------------------
# Controller stubs (your formulas)
# ----------------------------
def speed_mps(v: carla.Vector3D) -> float:
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

def pure_pursuit_steer(vehicle_tf: carla.Transform, target_loc: carla.Location, Ld: float, k_delta=1.5):
    loc = vehicle_tf.location
    yaw = math.radians(vehicle_tf.rotation.yaw)
    dx = target_loc.x - loc.x
    dy = target_loc.y - loc.y

    xv = math.cos(yaw)*dx + math.sin(yaw)*dy
    yv = -math.sin(yaw)*dx + math.cos(yaw)*dy

    if xv <= 1e-3:
        return 0.0
    kappa = 2.0 * yv / (Ld*Ld)
    delta = k_delta * kappa
    return max(-1.0, min(1.0, delta))

class SpeedPID:
    def __init__(self, kp=0.3, ki=0.05, kd=0.0, dt=0.05, i_clip=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.i_clip = i_clip
        self.i = 0.0
        self.prev_e = 0.0

    def step(self, v_target, v_curr):
        e = v_target - v_curr
        self.i = max(-self.i_clip, min(self.i_clip, self.i + e*self.dt))
        de = (e - self.prev_e) / self.dt
        self.prev_e = e
        return self.kp*e + self.ki*self.i + self.kd*de

# ----------------------------
# Main loop (you finish)
# ----------------------------
def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    tm = client.get_trafficmanager(8000)

    fixed_dt = 0.05
    set_sync(world, tm, fixed_dt)

    ego = spawn_ego(world, spawn_index=0)
    carla_map = world.get_map()

    # Pick destination spawn
    sps = carla_map.get_spawn_points()
    start_tf = sps[0]
    goal_tf = sps[min(10, len(sps)-1)]

    # Build graph
    nodes, adj = build_topology_graph(carla_map)
    print(f"graph: nodes={len(nodes)}")

    # Snap start/end to graph
    start_wp = carla_map.get_waypoint(start_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    goal_wp  = carla_map.get_waypoint(goal_tf.location,  project_to_road=True, lane_type=carla.LaneType.Driving)
    start_k = snap_to_graph_node(nodes, start_wp)
    goal_k  = snap_to_graph_node(nodes, goal_wp)

    # Plan (your A*)
    node_path = astar(nodes, adj, start_k, goal_k)
    print(f"node path length: {len(node_path)}")

    # Densify to route waypoints
    route_wps = densify_path(nodes, node_path, resolution_m=2.0)
    print(f"dense route waypoints: {len(route_wps)}")

    # Drive
    pid = SpeedPID(dt=fixed_dt)
    wp_idx = 0
    v_target = 10.0

    while True:
        world.tick()

        tf = ego.get_transform()
        v = speed_mps(ego.get_velocity())

        # lookahead Ld = clip(2+0.3*v, 5, 15)
        Ld = max(5.0, min(15.0, 2.0 + 0.3*v))

        # TODO: advance wp_idx toward lookahead target
        # - find a monotonic method (only increases)
        # - pick target waypoint route_wps[wp_idx_lookahead]

        # TODO: compute steer, throttle/brake
        # steer = pure_pursuit_steer(tf, target_loc, Ld)
        # u = pid.step(v_target, v)
        # map u to throttle/brake

        # TODO: ego.apply_control(...)
        # TODO: print/log every N ticks: v, wp_idx, distance to goal

if __name__ == "__main__":
    main()