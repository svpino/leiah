const admin = require("firebase-admin");
const artifacts = require("./artifacts");

/**
 * Returns the Firebase account associated with the specified uid or email. If both
 * the uid and email are specified, the uid will be used.
 *
 * @param {*} uid The uid of the user.
 * @param {*} email The email of the user.
 * @returns The Firebase account associated with the specified uid or email. Returns
 * null if an account is not found.
 */
exports.getExistingFirebaseAccount = async function (uid = null, email = null) {
    try {
        if (uid) {
            return await admin.auth().getUser(uid);
        }

        return await admin.auth().getUserByEmail(email);
    } catch (e) {
        // Firebase's getUserByEmail throws an exception if there's no account with the
        // specified email. If that happens, we just need to return null.
        return null;
    }
};

/**
 * Finds a user with the supplied email throughout all existing tenants and returns
 * a promise containing a QuerySnapshot. This query is possible thanks to the collection
 * group index created in Firestore.
 *
 * Keep in mind that this query will return the user document from the main /users collection
 * as well, so make sure you check for that in case it matters.
 *
 * @param {*} email The email of the user.
 */

exports.getUserAcrossAllTenants = async function (email) {
    return admin
        .firestore()
        .collectionGroup("users")
        .where("email", "==", email)
        .get();
};
