#!/usr/bin/env python3
"""
Query the orchestrator from the command line.

Usage:
  ./run_orchestrator_query.py "how many people are in the scene?"
  ./run_orchestrator_query.py "find the red cup"
  ./run_orchestrator_query.py "read the text on the cube"
  ./run_orchestrator_query.py "select the cube with a 2"

Publishes the query to /orchestrator/query, waits for a response
on /orchestrator/response, and prints the result.
"""
import sys
import time
import argparse

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import String as StringMsg


class OrchestratorQueryClient(Node):
    def __init__(self, query, timeout=10.0):
        super().__init__('orchestrator_query_client')
        self.query = query
        self.timeout = timeout
        self.response = None
        self.received = False

        self._sub = self.create_subscription(
            StringMsg, '/orchestrator/response', self._response_cb, 10)
        self._pub = self.create_publisher(
            StringMsg, '/orchestrator/query', 10)

    def _response_cb(self, msg):
        self.response = msg.data
        self.received = True

    def run(self):
        print(f'Query: {self.query}')
        sys.stdout.flush()

        # Let subscription settle, then publish
        time.sleep(0.2)
        out = StringMsg()
        out.data = self.query
        self._pub.publish(out)

        deadline = time.time() + self.timeout
        while rclpy.ok() and not self.received and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self.response:
            print(self.response)
        else:
            print(f'[orchestrator_query] No response within {self.timeout:.0f}s')
            return 1
        return 0


def main():
    parser = argparse.ArgumentParser(
        description='Query the orchestrator and print the response.')
    parser.add_argument('query', nargs='+', help='Text query')
    parser.add_argument('--timeout', type=float, default=10.0,
                        help='Seconds to wait for a response')
    args = parser.parse_args()
    query = ' '.join(args.query)

    rclpy.init()
    client = OrchestratorQueryClient(query, timeout=args.timeout)
    rc = client.run()
    client.destroy_node()
    rclpy.shutdown()
    sys.exit(rc)


if __name__ == '__main__':
    main()
