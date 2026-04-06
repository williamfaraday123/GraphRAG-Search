terraform {
  required_providers {
    alicloud = {
      source  = "aliyun/alicloud"
      version = "~> 1.0"
    }
  }
}

provider "alicloud" {
  region = "cn-beijing" # Change to your region
}

# ==========================================
# NETWORKING (Same as before)
# ==========================================
resource "alicloud_vpc" "main" {
  vpc_name   = "open-source-ai-vpc"
  cidr_block = "10.0.0.0/8"
}

resource "alicloud_vswitch" "main" {
  vswitch_name = "ai-subnet"
  cidr_block   = "10.0.1.0/24"
  vpc_id       = alicloud_vpc.main.id
  zone_id      = data.alicloud_zones.default.zones[0].id
}

data "alicloud_zones" "default" {
  available_resource_creation = "VSwitch"
}

# ==========================================
# SECURITY GROUP
# ==========================================
resource "alicloud_security_group" "ai_sg" {
  name   = "ai-open-source-sg"
  vpc_id = alicloud_vpc.main.id
}

# Allow inbound traffic for Web (80), SSH (22), and internal services
resource "alicloud_security_group_rule" "http" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "80/80"
  source_cidr_ip    = "0.0.0.0/0"
  security_group_id = alicloud_security_group.ai_sg.id
}

resource "alicloud_security_group_rule" "ssh" {
  type              = "ingress"
  ip_protocol       = "tcp"
  port_range        = "22/22"
  source_cidr_ip    = "0.0.0.0/0" # Restrict this to your IP in production
  security_group_id = alicloud_security_group.ai_sg.id
}

# Allow internal traffic between containers/microservices
resource "alicloud_security_group_rule" "internal" {
  type              = "ingress"
  ip_protocol       = "all"
  source_group_id   = alicloud_security_group.ai_sg.id
  security_group_id = alicloud_security_group.ai_sg.id
}

# ==========================================
# DATA & COMPUTE HOST (The Heavy Lifter)
# ==========================================
# Instead of managed DBs, we use one powerful ECS to run Docker containers
# Milvus and Neo4j need decent RAM and CPU.

resource "alicloud_instance" "db_and_compute" {
  instance_name          = "open-source-ai-host"
  image_id               = "ubuntu_22_04_x64_20G_alibase_20230710.vhd"
  instance_type          = "ecs.g7.4xlarge" # 16vCPU, 64GB RAM - Recommended for Milvus + Neo4j
  security_groups        = [alicloud_security_group.ai_sg.id]
  vswitch_id             = alicloud_vswitch.main.id
  # Attach a larger data disk for database storage
  system_disk_category   = "cloud_essd"
  system_disk_size       = 100
  data_disks {
    size = 500 # 500GB SSD for vector and graph data
    category = "cloud_essd"
  }

  # User Data: Bootstrap Docker and Docker-Compose
  user_data = <<-EOF
              #!/bin/bash
              set -e
  
              # 1. Install Docker
              apt-get update
              apt-get install -y docker.io docker-compose
  
              # 2. Add ubuntu user to docker group to run without sudo
              usermod -aG docker ubuntu
  
              # 3. Create a directory for the AI stack
              mkdir -p /opt/ai-search
              cat > /opt/ai-search/docker-compose.yml << 'EOT'
version: '3.5'

services:
  # 1. Milvus Standalone (Vector DB)
  milvus-standalone:
    image: milvusdb/milvus:v2.4
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
    volumes:
      - ./volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530" # Milvus SDK port
    depends_on:
      - etcd
      - minio

  # 2. Neo4j (Graph DB)
  neo4j:
    image: neo4j:5
    container_name: neo4j
    environment:
      - NEO4J_AUTH=neo4j/password123 # Change this!
      - dbms.security.procedures.unrestricted=*
    volumes:
      - ./volumes/neo4j/data:/data
      - ./volumes/neo4j/logs:/logs
    ports:
      - "7474:7474" # Browser
      - "7687:7687" # Bolt driver
    depends_on:
      - milvus-standalone

  # 3. MinIO (S3 Compatible Storage for source docs)
  minio:
    image: minio/minio
    container_name: minio
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - ./volumes/minio:/minio_data
    ports:
      - "9000:9000" # API
      - "9001:9001" # Console
    command: server /minio_data --console-address ":9001"

  # 4. Etcd (Required for Milvus)
  etcd:
    image: quay.io/coreos/etcd:v3.5
    container_name: etcd
    volumes:
      - ./volumes/etcd:/etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000

  # 5. MinIO (Object Storage for Milvus)
  minio:
    image: minio/minio
    container_name: minio
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin
    volumes:
      - ./volumes/minio:/minio_data
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /minio_data --console-address ":9001"

volumes:
  milvus_data:
  neo4j_data:
  minio_data:
EOT

              # 4. Start the services in the background
              cd /opt/ai-search && docker-compose up -d

              # 5. Ensure services start on boot
              echo "cd /opt/ai-search && docker-compose up -d" >> /etc/rc.local
              EOF
}

# ==========================================
# API GATEWAY (Spring Boot)
# ==========================================
# This remains mostly the same, but you might scale it down
resource "alicloud_instance" "api_gateway" {
  instance_name          = "springboot-api"
  image_id               = "ubuntu_22_04_x64_20G_alibase_20230710.vhd"
  instance_type          = "ecs.g7.large"
  security_groups        = [alicloud_security_group.ai_sg.id]
  vswitch_id             = alicloud_vswitch.main.id
  # ... (user_data to install Java/Spring Boot)
}

# ==========================================
# OUTPUTS
# ==========================================
output "database_host_ip" {
  value = alicloud_instance.db_and_compute.public_ip
}

output "neo4j_browser_url" {
  value = "http://${alicloud_instance.db_and_compute.public_ip}:7474"
  description = "Neo4j Browser Interface"
}

output "minio_console_url" {
  value = "http://${alicloud_instance.db_and_compute.public_ip}:9001"
  description = "MinIO Console (S3 Storage)"
}