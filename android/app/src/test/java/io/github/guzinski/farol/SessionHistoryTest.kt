package io.github.guzinski.farol

import io.github.guzinski.farol.perception.LampColor
import io.github.guzinski.farol.session.SavedAlert
import io.github.guzinski.farol.session.SessionHistory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

class SessionHistoryTest {
    @get:Rule val temp = TemporaryFolder()

    @Test
    fun readsExistingRunsAndDistinguishesEmptyLogsFromMissingFiles() {
        val previous = temp.newFolder("2026-11-18_09-12-01")
        val latest = temp.newFolder("2026-11-18_10-30-05")
        temp.newFolder("models")
        File(previous, "session.json").writeText("""
            {"mode":"drive","analysis_resolution":[3840,2160],"analysed_frames":2,
             "video_duration_s":1.2,"analysis_fps":15.0,"error":null}
        """.trimIndent())
        File(previous, "alerts.json").writeText("[]")
        File(previous, "detections.jsonl").writeText("""{"t":0.1,"detections":[]}""")

        val sessions = SessionHistory.list(temp.root)
        assertEquals(listOf(latest.name, previous.name), sessions.map { it.id })
        assertFalse(sessions.first().finalized)
        assertNull(SessionHistory.details(sessions.first()).alerts)
        val drive = sessions.last()
        assertEquals("3840 × 2160", drive.resolution)
        assertEquals("drive", drive.mode)
        assertNull(drive.error)
        assertEquals(emptyList<SavedAlert>(), SessionHistory.details(drive).alerts)
        assertEquals(1, SessionHistory.details(drive).detections?.frames)
    }

    @Test
    fun summarizesRepeatedTracksAndKeepsValidRowsFromPartialWrite() {
        val directory = temp.newFolder("2026-11-18_10-30-05")
        File(directory, "detections.jsonl").writeText("""
            {"t":0.1,"detections":[{"class":9,"track_id":1,"color":null},{"class":11,"track_id":2}]}
            {"t":0.2,"detections":[{"class":9,"track_id":1,"color":"red"},{"class":9,"track_id":3,"color":"green"}]}
            {"t":0.3,"detections":[]}
            {"t":0.4,"detections":[
        """.trimIndent())
        File(directory, "alerts.json").writeText("""
            [{"t":2.5,"text":"Semáforo vermelho"},{"t":0.6,"text":"Placa de pare à frente"}]
        """.trimIndent())

        val details = SessionHistory.details(SessionHistory.list(temp.root).single())
        val detections = checkNotNull(details.detections)
        assertEquals(3, detections.frames)
        assertEquals(2, detections.lightTracks)
        assertEquals(1, detections.stopTracks)
        assertEquals(3, detections.lightObservations)
        assertEquals(mapOf(LampColor.RED to 1, LampColor.GREEN to 1), detections.colors)
        assertEquals(1, detections.unreadableLines)
        assertEquals(listOf(0.6, 2.5), details.alerts?.map { it.seconds })
    }
}
