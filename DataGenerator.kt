package pl.lipov.algorytmy

import android.content.Context
import android.util.Log
import pl.lipov.algorytmy.Repository.floors
import pl.lipov.algorytmy.Repository.landmarkPositions
import pl.lipov.algorytmy.Repository.landmarks
import pl.lipov.algorytmy.Repository.rooms
import pl.lipov.algorytmy.Repository.userPositions
import java.io.File

object DataGenerator {

    private const val TAG = "DataGenerator"

    fun generate(context: Context) {
        val datasetSize = 1000
        val file = File(context.getExternalFilesDir(null), "geo_dataset_input.txt")

        file.printWriter().use { writer ->
            repeat(datasetSize) {
                writer.println(generateInputLine())
            }
        }

        Log.d(TAG, "Wygenerowano $datasetSize inputów i zapisano do geo_dataset_input.txt")
    }

    private fun generateInputLine(): String {
        val floor = floors.random()
        val room = rooms.random()
        val landmark = landmarks.random()
        val userPosition = userPositions.random()
        val landmarkPosition = landmarkPositions.random()
        val includeLandmarkInfo = (0..1).random() == 1

        return buildString {
            append("geoDescriptionType: location; ")
            append("floor: $floor; ")
            append("userPositionInfo: $userPosition $room; ")

            if (includeLandmarkInfo) {
                append("landmarkPositionInfo: $landmark $landmarkPosition; ")
            }
        }
    }
}
