import SwiftUI

@main
struct TeawaseApp: App {
    var body: some Scene {
        WindowGroup {
            WebGameView()
                .ignoresSafeArea()
                .background(Color(red: 0.03, green: 0.02, blue: 0.12))
                .statusBarHidden(true)
        }
    }
}
