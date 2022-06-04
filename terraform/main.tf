terraform {
  backend "gcs" {
    bucket = "birdclef-2022-tfstate"
  }
}

locals {
  project_id = "birdclef-2022"
  region     = "us-central1"
}

provider "google" {
  project = local.project_id
  region  = local.region
}

resource "google_storage_bucket" "birdclef-2022" {
  name     = local.project_id
  location = "US"
  versioning {
    enabled = true
  }
  lifecycle_rule {
    condition {
      num_newer_versions = 3
    }
    action {
      type = "Delete"
    }
  }
  cors {
    origin          = ["*"]
    method          = ["GET"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

resource "google_storage_bucket_iam_binding" "default-public" {
  bucket = google_storage_bucket.birdclef-2022.name
  role   = "roles/storage.objectViewer"
  members = [
    "allUsers"
  ]
}

output birdclef-2022-bucket {
  value = google_storage_bucket.birdclef-2022.url
}
