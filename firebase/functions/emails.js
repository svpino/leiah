const functions = require("firebase-functions");
const admin = require("firebase-admin");
const { PubSub } = require("@google-cloud/pubsub");

const sendgrid = require("@sendgrid/mail");
const constants = require("./constants");

/**
 * This function will send an invitation email to a user that was invited to a tenant.
 *
 * @param {*} tenant The id of the tenant where the user was invited.
 * @param {*} email The email of the user that was invited.
 * @param {*} role The role of the user. Most likely this role will be "invited", but it might be the case that
 * this function gets called for a user that was manually added to a tenant (probably because we were testing
 * things) and its role is not invited, in which case this method will do nothing.
 * @param {*} name The name of the user that was invited.
 * @param {*} invitation The invitation information created with the invite.
 */
exports.sendUserInvitation = async function (tenant, email, role, name, invitation) {
    // We should only send an email invitation to those users that were invited to a tenant. Remember that users
    // will be added to tenants during registration as well.
    if (role === "invited") {
        // We need to get the name of the tenant because it will go as part of the invitation email.
        admin
            .firestore()
            .doc("tenants/" + tenant)
            .get()
            .then(async (tenantSnapshot) => {
                tenantName = tenantSnapshot.data().name;

                const pubsub = new PubSub();
                const attributes = {
                    template: functions.config().sendgrid.templates.user_invitation,
                    email: email,
                    name: name,
                    company: tenantName,
                    link: `${functions.config().general.product_url}/link`,
                };

                // The personalized message is not mandatory, so we need to make sure we handle it appropriately.
                invitation_message =
                    invitation && invitation.message ? invitation.message : "";

                const dataBuffer = Buffer.from(invitation_message);
                const messageId = await pubsub
                    .topic(
                        `projects/${admin.instanceId().app.options.projectId}/topics/${
                            functions.config().sendgrid.pubsub_topic
                        }`
                    )
                    .publish(dataBuffer, attributes);
                return functions.logger.info(
                    `Published message ${messageId} to invite user ${email} to tenant ${tenant}.`
                );
            })
            .catch((error) => {
                return functions.logger.error(
                    `Error publishing message to invite user ${email} to tenant ${tenant}. ${error}.`
                );
            });
    }
};

/**
 * This function is connected to a Pub/Sub topic and it's responsibility is to read the data posted,
 * construct an email payload, fill out the specific template attributes, and send the email notification.
 */
exports.onPublishEmail = functions.pubsub.topic("emails").onPublish((message) => {
    sendgrid.setApiKey(functions.config().sendgrid.key);

    template = message.attributes.template;

    const payload = {
        to: message.attributes.email,
        from: functions.config().sendgrid.from, 
        templateId: template,
    };

    if (template === constants.EMAIL_TEMPLATE_EMAIL_VERIFICATION) {
        payload.dynamic_template_data = {
            name: message.attributes.name,
            link: message.attributes.link,
        };
    } else if (template === constants.EMAIL_TEMPLATE_IDENTITY_VERIFICATION) {
        payload.dynamic_template_data = {
            name: message.attributes.name,
            code: message.attributes.code,
        };
    } else if (template === functions.config().sendgrid.templates.user_invitation) {
        payload.dynamic_template_data = {
            name: message.attributes.name,
            company: message.attributes.company,
            link: message.attributes.link,
        };

        const content = message.data
            ? Buffer.from(message.data, "base64").toString()
            : null;
        if (content) {
            payload.message = content;
        }
    } else if (template === constants.EMAIL_TEMPLATE_PASSWORD_RESET) {
        payload.dynamic_template_data = {
            name: message.attributes.name,
            link: message.attributes.link,
        };
    } else {
        console.error(`The specified template ${template} is not valid.`);
        return null;
    }

    return sendgrid
        .send(payload)
        .then(() => {
            return functions.logger.info(
                `An email notification with template ${template} was sent to ${message.attributes.email}.`
            );
        })
        .catch((error) => {
            // If this was a SendGrid error, let's make sure we dive into the specific of the error because
            // Firebase logs don't expand messages too far from the top.
            if (error.response && error.response.body) {
                error = error.response.body;
            }

            return functions.logger.error(
                `Error sending email notification with template ${template} to ${message.attributes.email}. ${error}`
            );
        });
});
