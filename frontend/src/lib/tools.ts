/**
 * The developer tools of the stack that an administrator can open from the dashboard.
 * Host ports are the defaults of docker-compose.yml (MLFLOW_PORT, QDRANT_PORT, ADMINER_PORT,
 * SPARK_UI_PORT); tests/test_docker_compose.py checks that both files agree.
 */
export const TOOL_PORTS = {
  mlflow: 5000,
  qdrant: 6333,
  adminer: 8081,
  spark: 8090,
} as const;

/** What to type in the Adminer login form (POSTGRES_USER and the databases of compose). No password. */
export const ADMINER_LOGIN = {
  system: "PostgreSQL",
  server: "postgres",
  username: "tujya",
  databases: ["tujya", "mlflow"],
} as const;

export type ToolId = "mlflow" | "qdrant" | "adminer" | "swagger" | "spark";

export interface Tool {
  id: ToolId;
  url: string;
}

/**
 * The address of each tool on the machine that serves the app. Swagger goes through the
 * frontend (nginx forwards /docs to the API); the others are on their own published port.
 */
export function toolLinks(hostname: string, ports: Record<keyof typeof TOOL_PORTS, number> = TOOL_PORTS): Tool[] {
  // An IPv6 literal needs brackets in a URL.
  const host = hostname.includes(":") ? `[${hostname}]` : hostname;
  const at = (port: number, path = "") => `http://${host}:${port}${path}`;
  return [
    { id: "mlflow", url: at(ports.mlflow) },
    { id: "qdrant", url: at(ports.qdrant, "/dashboard") },
    { id: "adminer", url: at(ports.adminer) },
    { id: "swagger", url: "/docs" },
    { id: "spark", url: at(ports.spark) },
  ];
}
