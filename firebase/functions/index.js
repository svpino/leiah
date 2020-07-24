const functions = require("firebase-functions");
const admin = require("firebase-admin");

admin.initializeApp(functions.config().firebase);

const artifacts = require("./artifacts");

/**
 * This function triggers whenever a user document from the main /users
 * collection is deleted. It removes the associated Firebase account and
 * any existent reference to this user within the /tenants collection.
 */
exports.onUserDelete = functions.firestore
    .document("users/{userId}")
    .onDelete(async (snap, context) => {
        const email = context.params.userId;
        functions.logger.info(`User ${email} removed from the database.`);

        return await Promise.all([
            deleteUserAccountFromFirebase(snap.data().uid, email),
            deleteUserFromAllExistingTenants(email),
        ]);
    });

/**
 * This function triggers whenever a user document from the main /users
 * collection is updated. It keeps the references to this user updated.
 */
exports.onUserUpdate = functions.firestore
    .document("users/{userId}")
    .onUpdate(async (change, context) => {
        const oldDocument = change.before.exists ? change.before.data() : null;
        const newDocument = change.after.exists ? change.after.data() : null;
        const email = context.params.userId;
        const provider = newDocument.provider;

        if (provider && oldDocument.provider !== provider) {
            const querySnapshot = await artifacts.getUserAcrossAllTenants(email);

            for (let userSnapshot of querySnapshot.docs) {
                updateUserSnapshotOnTenant(userSnapshot, email, provider);
            }
        }
    });

/**
 * This function triggers whenever a new user is added to the tenant's users
 * collection. This function creates a user document in the main /users collection
 * if it doesn't exist already and to send an email invitation to the new user.
 */
exports.onTenantUserCreate = functions.firestore
    .document("tenants/{tenantId}/users/{userId}")
    .onCreate(async (snap, context) => {
        const email = context.params.userId;
        const tenant = context.params.tenantId;
        const name = snap.data().name;
        const role = snap.data().role;

        let invitation = role === "invited" ? snap.data().invitation : null;

        functions.logger.info(`User ${email} added to tenant ${tenant}.`);

        // Keep in mind that creating the user in the main collection can fail but
        // we may have sent the invitation email. This means that there's a chance
        // for a user to accept an invitation and not have a corresponding user
        // document created in the main /users collection.
        return await Promise.all([
            createInvitedUser(email, name),
            // notifications.sendUserInvitation(tenant, email, role, name, invitation),
        ]);
    });

/**
 * Creates a new user document in the main /users collection whenever a new user
 * is invited to a tenant. We want to create this user document in the main
 * collection regardless of the role of the added user.
 *
 * @param {*} email The email of the user that was invited.
 * @param {*} name The name of the user that was invited.
 */
async function createInvitedUser(email, name) {
    admin
        .firestore()
        .doc("users/" + email)
        .get()
        .then((userSnapshot) => {
            if (!userSnapshot.exists) {
                return createInvitedUserInMainCollection(email, name);
            }

            return null;
        })
        .catch((error) => {
            return functions.logger.error(
                `Error loading user document representing ${email} from main /users collection. ${error}`
            );
        });
}

async function createInvitedUserInMainCollection(email, name) {
    admin
        .firestore()
        .doc("users/" + email)
        .create({
            email: email,
            name: name,
            uid: null,
        })
        .then(() => {
            return functions.logger.info(
                `User document representing ${email} created in main /users collection`
            );
        })
        .catch((error) => {
            return functions.logger.error(
                `Error creating user document representing ${email} in main /users collection. ${error}`
            );
        });
}

/**
 * Deletes a user account from Firebase's authentication.
 *
 * @param {*} uid The uid field of the user account in Firebase. This field
 * could be undefined.
 * @param {*} email The email of the user.
 */
async function deleteUserAccountFromFirebase(uid, email) {
    // If the uid attribute is not defined, we need to find the Firebase account
    // using the email address.
    if (!uid) {
        let firebaseAccount = await artifacts.getExistingFirebaseAccount(null, email);
        uid = firebaseAccount && firebaseAccount.uid ? firebaseAccount.uid : null;
    }

    if (uid) {
        return admin
            .auth()
            .deleteUser(uid)
            .then(() => {
                return functions.logger.info(
                    `Firebase account with uid ${uid} has been successfully deleted.`
                );
            })
            .catch((error) => {
                if (error.code !== "auth/user-not-found") {
                    return functions.logger.error(
                        `Error deleting Firebase account with uid ${uid}. ${error}`
                    );
                }

                return functions.logger.debug(`User ${uid} doesn't exist in Firebase.`);
            });
    }

    return functions.logger.debug(`User ${email} doesn't exist in Firebase.`);
}

/**
 * Deletes a user from all existing tenants (from the /tenants/users/ sub-collection}).
 * This function helps when a user document is removed from the main /users collection.
 *
 * @param {*} email The email of the user.
 */
async function deleteUserFromAllExistingTenants(email) {
    artifacts
        .getUserAcrossAllTenants(email)
        .then((querySnapshot) => {
            querySnapshot.forEach((userSnapshot) => {
                return deleteUserSnapshotFromTenant(userSnapshot);
            });

            return functions.logger.info(
                `User ${email} has been successfully removed from all existing tenants.`
            );
        })
        .catch((error) => {
            return functions.logger.error(
                `There was an error retrieving user ${email} from all existing tenants. ${error}`
            );
        });
}

/**
 * Updates the specified user snapshot from a tenant with the supplied information.
 *
 * @param {*} userSnapshot The user snapshot that should be updated.
 * @param {*} provider The provider attribute.
 */
async function updateUserSnapshotOnTenant(userSnapshot, email, provider) {
    if (userSnapshot.ref.path.startsWith("tenants/")) {
        try {
            await userSnapshot.ref.set(
                {
                    provider: provider,
                },
                { merge: true }
            );

            functions.logger.info(
                `Updated user ${email} from tenant ${userSnapshot.ref.parent.parent.id}.`
            );
        } catch (error) {
            functions.logger.error(
                `There was an error updating user ${email} across all existing tenants. ${error}`
            );
        }
    }
}

/**
 * Deletes the specified user snapshot from a tenant.
 *
 * @param {*} userSnapshot The user snapshot that should be deleted.
 */
async function deleteUserSnapshotFromTenant(userSnapshot) {
    if (userSnapshot.ref.path.startsWith("tenants/")) {
        userSnapshot.ref
            .delete()
            .then(() => {
                return functions.logger.info(
                    `Removed user ${email} from tenant ${userSnapshot.ref.parent.parent.id}`
                );
            })
            .catch((error) => {
                return functions.logger.error(
                    `There was an error deleting user ${email} from from tenant ${userSnapshot.ref.parent.parent.id}. ${error}`
                );
            });
    }

    return null;
}
