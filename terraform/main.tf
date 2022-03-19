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
}

output birdclef-2022-bucket {
  value = google_storage_bucket.birdclef-2022.url
}
