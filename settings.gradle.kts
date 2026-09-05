pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google() // REQUIRED for play-services-ads
        mavenCentral()
    }
}
rootProject.name = "MagicImagePro"
include(":app")
